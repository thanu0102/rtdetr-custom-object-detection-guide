# RT-DETR Custom Object Detection Guide

An end-to-end tutorial for training and deploying RT-DETR on custom datasets.

## Overview

This project demonstrates the complete workflow for object detection using RT-DETR, including:

- Dataset preparation
- Annotation workflow
- Training RT-DETR models
- Evaluation using common metrics
- Running inference on new images
- Visualizing predictions

## Project Structure

```text
rtdetr-custom-object-detection-guide
│
├── README.md
├── requirements.txt
│
├── docs
│   ├── dataset_preparation.md
│   ├── training.md
│   └── inference.md
│
├── examples
│   ├── inference.py
│   └── visualize_predictions.py
│
├── images
│
└── notebooks
    └── RTDETR_Tutorial.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

## Topics Covered

- RT-DETR architecture
- Custom dataset creation
- Training pipeline
- Evaluation metrics
- Deployment workflow

## Documentation

- Understanding RT-DETR Architecture
- Dataset Preparation Guide
- Training Guide
- Inference Guide

## Tutorial Notebook

The complete workflow is available in:

`notebooks/RTDETR_Tutorial.ipynb`

## Future Improvements

- Video inference
- Model comparison with YOLO
- Performance benchmarking

## References

- RT-DETR Paper
- Ultralytics Documentation

