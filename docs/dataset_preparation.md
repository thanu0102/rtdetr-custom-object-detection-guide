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

## Annotation Format

Annotations must be saved as `.txt` files (one per image) in YOLO format: <class_id> <x_center> <y_center> <width> <height>
   
Where coordinates are normalized (0-1) relative to image dimensions.

Example `image.txt`:
```
0 0.45 0.50 0.30 0.40

1 0.75 0.65 0.15 0.25
```

## Dataset Split

The dataset should be divided into:

* Training Set (60-70%)
* Validation Set (10–20%)
* Test Set (10–20%)

This helps evaluate how well the model generalizes to unseen data.

## Dataset Structure

The dataset should use the Horizontal Bounding Box (HBB) format. Other formats (rotated boxes, polygons) are not supported and will cause training failures.

Example directory structure:

```
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

## Dataset Configuration

A dataset.yaml file is used to define:

* Dataset paths
* Number of classes
* Class names
* Augmentations

Example:
```
path: /path/to/dataset
   train: images/train
   val: images/val
   test: images/test
   
   nc: 4  # Number of classes
   names: ['person', 'vehicle', 'road', 'pedestrian']
   
   # Augmentations (optional for RT-DETR)
   degrees: 10.0
   translate: 0.1
   scale: 0.5
   flipud: 0.0
   fliplr: 0.5
```

## Conclusion

Proper data collection, annotation, and dataset organization are essential for obtaining high-quality RT-DETR detection results.

## Note

RT-DETR uses the same YOLO-format dataset structure as YOLO, making it easy to switch between detectors without reformatting your data.
