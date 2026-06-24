# Dataset Preparation for RT-DETR

## Introduction

The quality of an object detection model depends heavily on the quality of the dataset. Before training RT-DETR, images must be collected, annotated, and organized into a suitable format.

## Data Collection

Data can be collected from:

* Videos captured from cameras or drones
* Public datasets
* Internet sources
* Existing image repositories

During real-world object detection projects, video footage is often converted into frames and manually reviewed before annotation. Removing blurry and redundant frames improves dataset quality and reduces labeling effort.

## Frame Extraction

Videos are split into individual frames to create a training dataset.

Typical workflow:

Video → Frames → Annotation → Training Dataset

When extracting frames, it is important to avoid excessive duplication by selecting frames at regular intervals.

## Annotation

Each object of interest must be labeled using a bounding box.

Common annotation tools include: 

* CVAT
* LabelImg
* Roboflow

(I used CVAT)

For every image:

* Draw a bounding box around the object.
* Assign the correct class label.
* Save annotations in YOLO format.

## Dataset Split

The dataset should be divided into:

* Training Set (60-70%)
* Validation Set (10–20%)
* Test Set (10–20%)

This helps evaluate how well the model generalizes to unseen data.

## Dataset Structure

The dataset must use the Horizontal Bounding Box (HBB) format. Using any other format will lead to:
* Poor model accuracy
* Incorrect evaluation results
* Complete failure of the training methodology 

Example directory structure:

dataset/

├── train/

│   ├── images/

│   ├── labels/

└── val/

│   ├── images/

│   ├── labels/

└── test/

│   ├── images/

│   ├── labels/

```
├── train/

├── val/

└── test/
```

## Dataset Configuration

A dataset.yaml file is used to define:

* Dataset paths
* Number of classes
* Class names
* Augmentations

Example:

path: dataset
train: images/train
val: images/val

names:
0: person
1: vehicle
2: road
3: pedestrian 

degrees: 10.0      # Rotate image by +/- 10 degrees
translate: 0.1     # Translate image horizontally/vertically by +/- 10%
scale: 0.5         # Scale image by +/- 50%
shear: 2.0         # Shear image by +/- 2 degrees
flipud: 0.0        # Probability of flipping image upside down
fliplr: 0.5  

## Conclusion

Proper data collection, annotation, and dataset organization are essential for obtaining high-quality RT-DETR detection results.
