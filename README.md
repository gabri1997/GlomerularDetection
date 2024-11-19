# Glomeruli Detection

This repository contains a project aimed at detecting glomeruli in histopathological images using deep learning techniques. The goal is to build an automated system for identifying and segmenting glomeruli regions in tissue slide images, which is useful for various medical applications, including kidney disease diagnosis.

## Project Overview

The project uses **YOLOv8**, a state-of-the-art object detection model, to identify glomeruli in large-scale Whole Slide Images (WSI). The dataset used for training and validation contains annotations in the form of bounding boxes around glomeruli regions. The detection model is trained on these annotated data, and the output includes bounding boxes that mark the predicted glomeruli locations.

## Key Features
- **YOLOv8 Model**: Utilizes the YOLOv8 architecture for fast and accurate object detection.
- **Preprocessing**: Handles the segmentation and patching of WSIs for training purposes.
- **Custom Annotations**: Annotations in YOLO format are used for training and evaluation.
- **Prediction Visualization**: Draws bounding boxes around detected glomeruli and visualizes the results.

- ### Notes:
1. **`![Glomerulus Detection Example](path/to/your/image.jpg)`**
