# RT-DETR Inference Guide

This will be much shorter and easier.

## Loading a Trained Model

from ultralytics import RTDETR

model = RTDETR("best.pt")

1) Running Inference on an Image
results = model.predict(
    source="test.jpg",
    save=True
)

2) Running Inference on a Video
results = model.predict(
    source="video.mp4",
    save=True
)

## Output

The model generates: 

* Bounding boxes
* Class labels
* Confidence scores

## Practical Notes

Inference speed depends on:

* GPU availability
* Image resolution
* Model size

Lower resolutions generally improve throughput while higher resolutions may improve detection accuracy.


