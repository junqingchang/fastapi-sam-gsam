from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List
import cv2
import numpy as np
from contextlib import asynccontextmanager
from groundingdino.util.inference import (
    Model,
)
from segment_anything import sam_model_registry, SamPredictor
import torch
import torchvision
import os
import shutil
import zipfile
import uuid
import time

DEVICE_STRING = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STRING)

models = {}
BOX_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# Lifespan handler to create the model when the app starts and clean up on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application...")
    # Building GroundingDINO inference model
    grounding_dino_model = Model(
        model_config_path="configs/groundingdino.py",
        model_checkpoint_path="models/groundingdino_swint_ogc.pth",
        device=DEVICE_STRING,
    )

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    models["gdino"] = (
        grounding_dino_model  # Store the model in app.state for access in requests
    )
    models["sam"] = sam_predictor
    print("Model created")

    yield

    # Perform any cleanup here if necessary (e.g., freeing resources)
    print("Shutting down application...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello! Please visit /docs for api specs"}


# Function to delete the temporary directory in the background
def cleanup_directory(directory: str):
    shutil.rmtree(directory)


# Function to read the file with OpenCV
def read_image_with_cv2(file: UploadFile):
    file_bytes = np.frombuffer(file.file.read(), np.uint8)  # Convert file to bytes
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode image with OpenCV
    return image


# SAM segment function
def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# GSAM endpoint with image and text input
@app.post(
    "/gsam",
    summary="Process image and return results in a ZIP file",
    description="Processes the input image using GroundingDINO and SAM models, and returns the results (bounding boxes and masks) in a zip file.",
    response_class=FileResponse,
)
async def gsam(
    image_file: UploadFile = File(..., description="The image file to be processed"),
    text_prompt: str = Form(
        ..., description="Comma-separated list of class names to detect OR single words"
    ),
    threshold: float = Form(
        0.4, description="Detection threshold value (default: 0.4)"
    ),
    multimask: bool = Form(
        True,
        description="Boolean flag to indicate multi-detection logic (default: True)",
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    # Create a unique directory name using a combination of timestamp and UUID
    folder_name = f"gsam_output_{int(time.time())}_{uuid.uuid4()}"
    temp_dir = os.path.join(
        os.getcwd(), folder_name
    )  # Current working directory + unique folder
    os.makedirs(temp_dir, exist_ok=True)  # Create the directory

    try:
        # Read the image directly using OpenCV
        image = read_image_with_cv2(image_file)

        classes = text_prompt.split(",")

        # detect objects
        detections = models["gdino"].predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=threshold,
            text_threshold=threshold,
        )

        # NMS post process
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        output_bbox = []
        for bbox in detections.xyxy:
            bbox_string_list = [str(x) for x in bbox] + ["\n"]
            bbox_string = " ".join(bbox_string_list)
            output_bbox.append(bbox_string)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=models["sam"],
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        bbox_file_path = os.path.join(temp_dir, "output_bbox.txt")
        with open(bbox_file_path, "w") as f:
            f.writelines(output_bbox)

        # Save each mask as a PNG file
        for idx, mask in enumerate(detections.mask):
            mask_path = os.path.join(temp_dir, f"mask_{idx}.png")
            cv2.imwrite(mask_path, mask * 255)

        # Create a zip file to bundle all the outputs
        zip_filename = "gsam_output.zip"
        zip_filepath = os.path.join(temp_dir, zip_filename)

        with zipfile.ZipFile(zip_filepath, "w") as zipf:
            # Add the bounding box file
            zipf.write(bbox_file_path, arcname="output_bbox.txt")

            # Add mask PNG files to the zip
            for idx, mask in enumerate(detections.mask):
                mask_path = os.path.join(temp_dir, f"mask_{idx}.png")
                zipf.write(mask_path, arcname=f"mask_{idx}.png")

        # Schedule the directory for deletion in the background after response is sent
        background_tasks.add_task(cleanup_directory, temp_dir)

        # Return the zip file as the response
        return FileResponse(
            zip_filepath, media_type="application/zip", filename=zip_filename
        )
    except Exception as e:
        # If an error occurs, cleanup the directory immediately
        shutil.rmtree(temp_dir)
        return {"message": e}


# GDINO endpoint with image and text input
@app.post(
    "/gdino",
    summary="Process image and return results in a ZIP file",
    description="Processes the input image using GroundingDINO, and returns the results (bounding boxes) in a zip file.",
    response_class=FileResponse,
)
async def gdino(
    image_file: UploadFile = File(..., description="The image file to be processed"),
    text_prompt: str = Form(
        ..., description="Comma-separated list of class names to detect OR single words"
    ),
    threshold: float = Form(
        0.4, description="Detection threshold value (default: 0.4)"
    ),
    multimask: bool = Form(
        True,
        description="Boolean flag to indicate multi-detection logic (default: True)",
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    # Create a unique directory name using a combination of timestamp and UUID
    folder_name = f"gdino_output_{int(time.time())}_{uuid.uuid4()}"
    temp_dir = os.path.join(
        os.getcwd(), folder_name
    )  # Current working directory + unique folder
    os.makedirs(temp_dir, exist_ok=True)  # Create the directory
    try:
        # Read the image directly using OpenCV
        image = read_image_with_cv2(image_file)

        classes = text_prompt.split(",")

        # detect objects
        detections = models["gdino"].predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=threshold,
            text_threshold=threshold,
        )

        # NMS post process
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        output_bbox = []
        for bbox in detections.xyxy:
            bbox_string_list = [str(x) for x in bbox] + ["\n"]
            bbox_string = " ".join(bbox_string_list)
            output_bbox.append(bbox_string)

        bbox_file_path = os.path.join(temp_dir, "output_bbox.txt")
        with open(bbox_file_path, "w") as f:
            f.writelines(output_bbox)

        # Create a zip file to bundle all the outputs
        zip_filename = "gdino_output.zip"
        zip_filepath = os.path.join(temp_dir, zip_filename)

        with zipfile.ZipFile(zip_filepath, "w") as zipf:
            # Add the bounding box file
            zipf.write(bbox_file_path, arcname="output_bbox.txt")

        # Schedule the directory for deletion in the background after response is sent
        background_tasks.add_task(cleanup_directory, temp_dir)
        # Return the zip file as the response
        return FileResponse(
            zip_filepath, media_type="application/zip", filename=zip_filename
        )
    except Exception as e:
        # If an error occurs, cleanup the directory immediately
        shutil.rmtree(temp_dir)
        return {"message": e}


# SAM endpoint with image and list of values
@app.post("/sam")
async def sam(
    image_file: UploadFile = File(..., description="The image file to be processed"),
    point_prompt: List[List] = Form(
        ..., description="Coordinates of points to be used for point prompting"
    ),
    multimask: bool = Form(
        True,
        description="Boolean flag to indicate multi-detection logic (default: True)",
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    # Create a unique directory name using a combination of timestamp and UUID
    folder_name = f"sam_output_{int(time.time())}_{uuid.uuid4()}"
    temp_dir = os.path.join(
        os.getcwd(), folder_name
    )  # Current working directory + unique folder
    os.makedirs(temp_dir, exist_ok=True)  # Create the directory

    try:
        # Read the image directly using OpenCV
        image = read_image_with_cv2(image_file)
        models["sam"].set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        masks, scores, logits = models["sam"].predict(
            point_coords=np.array(point_prompt),
            point_labels=np.array(point_prompt),
            multimask_output=False,
        )
        index = np.argmax(scores)
        result_masks = [masks[index]]

        # Save each mask as a PNG file
        for idx, mask in enumerate(result_masks):
            mask_path = os.path.join(temp_dir, f"mask_{idx}.png")
            cv2.imwrite(mask_path, mask * 255)

        # Create a zip file to bundle all the outputs
        zip_filename = "sam_output.zip"
        zip_filepath = os.path.join(temp_dir, zip_filename)

        with zipfile.ZipFile(zip_filepath, "w") as zipf:
            # Add mask PNG files to the zip
            for idx, mask in enumerate(result_masks):
                mask_path = os.path.join(temp_dir, f"mask_{idx}.png")
                zipf.write(mask_path, arcname=f"mask_{idx}.png")

        # Schedule the directory for deletion in the background after response is sent
        background_tasks.add_task(cleanup_directory, temp_dir)

        # Return the zip file as the response
        return FileResponse(
            zip_filepath, media_type="application/zip", filename=zip_filename
        )
    except Exception as e:
        # If an error occurs, cleanup the directory immediately
        shutil.rmtree(temp_dir)
        return {"message": e}
