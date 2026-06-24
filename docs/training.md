# Training RT-DETR on a Custom Dataset

## Introduction

RT-DETR (Real-Time Detection Transformer) is an object detection model that combines the accuracy of Transformer-based detectors with real-time inference speed.

This guide demonstrates how to train RT-DETR on a custom dataset using the Ultralytics implementation.

## Prerequisites

Before training:

* Dataset must be annotated.
* Dataset should follow the YOLO directory structure and must in HBB format.
* A valid dataset.yaml file must be created.

## Installing Dependencies

```bash
pip install ultralytics
```

## Loading a Pretrained RT-DETR Model

```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
```

Using pretrained weights helps the model converge faster and achieve better performance.

## Training

```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16, 
    patience=20,
    device="cpu", 
    device=[0,1,2,3]  #Can use this if GPU's are available
)
```

## Important Hyperparameters

### Epochs

Defines how many times the model sees the entire training dataset.

Example:

```python
epochs=50
```

### Image Size

Controls the input image resolution.

Example:

```python
imgsz=640
```

Higher resolutions may improve accuracy but require more GPU memory.

### Batch Size

Controls the number of images processed simultaneously.

Example:

```python
batch=16
```

## Evaluation Metrics

### mAP (Mean Average Precision)

Measures overall detection performance.

### Precision

Measures how many predicted detections are correct.

### Recall

Measures how many true objects are successfully detected.

## Monitoring Training

During training, monitor:

* Training loss
* Validation loss
* mAP
* Precision
* Recall

The best model weights are automatically saved.

## Output Files

Typical outputs include:

* best.pt
* last.pt
* training logs
* metric plots
* sample predictions with bounding boxes

## Conclusion

Training RT-DETR on a high-quality dataset with appropriate hyperparameters can produce accurate and efficient object detection models suitable for real-world applications.
