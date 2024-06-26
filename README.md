weights: [https://drive.google.com/drive/folders/17Z7pwIn33e0r-HPkYyePeAdUvmtzWVai?usp=sharing](https://drive.google.com/drive/folders/17Z7pwIn33e0r-HPkYyePeAdUvmtzWVai?usp=sharing)


# Documentation for the "LIVECell Instance Segmentation Project"

## Variant 1: Detection using YOLOv8x and segmentation using OpenCV

### Project Structure
1. **data_prepare.py**: Script to convert COCO format data to YOLO format.
2. **train_model.py**: Script for model training and evaluation on the test dataset.
3. **inference.py**: Script to output two resulting images for each specified image: one with detection and another with segmentation using OpenCV.
4. **detection_train_notebook.ipynb**: Notebook detailing the training process conducted on Kaggle.

### Usage
1. Ensure that paths to files and directories in the scripts are correctly specified.
2. Download the pretrained model weights (links to weights are provided in the respective files).

## Variant 2: Instance segmentation using YOLOv8x-seg

### Project Structure
1. **data_prepare.py**: Script to convert COCO format data to YOLO format using a third-party library (installation function is provided in the file).
2. **train_model.py**: Script for model training.
3. **inference.py**: Script to output the instance segmentation result for each specified image.
4. **segmentation_train_notebook.ipynb**: Notebook detailing the training process conducted on Kaggle.

### Usage
1. Ensure that paths to files and directories in the scripts are correctly specified.
2. Download the pretrained model weights (links to weights are provided in the respective files).
