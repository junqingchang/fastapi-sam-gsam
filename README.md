# FastAPI Endpoint for using GSAM, GDINO, SAM
A bbox labeller built on 

[facebookresearch's segment anything](https://github.com/facebookresearch/segment-anything)

[IDEA-Research's GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

[IDEA-Research's GroundingSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)

## Directory Structure
```
configs/
    groundingdino.py
model/
    <segment anything models goes here>
.gitignore
main.py
README.md
requirements.txt
```

## Download Model Checkpoint
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Run FastAPI
```
$ fastapi run
```

# Grounded SAM
Refer to https://github.com/IDEA-Research/Grounded-Segment-Anything for more detailed information
```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

pip install -e Grounded-Segment-Anything/GroundingDINO

```
