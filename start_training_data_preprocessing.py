import config as config
import torch
import os
import sys
import torch.optim as optim
import matplotlib.pyplot as plt
from model.model_medium import YOLOv3
from tqdm import tqdm
from datetime import datetime
from src.utils.logger import Tee
from src.training.loss import YoloLoss
from src.preprocessing.preprocess_data import (
    process_label_cloud_labels_to_yolo_format,
    transform_to_bev_training_data,
    create_new_dataset
)

def main():
    """
    Main function to transform PCD files to BEV images and labelCloud labels to YOLO format labels.
    """
    # Define dataset paths
    dataset_path = os.path.join("model_training_data", "datasets", config.NEW_DATASET_NAME)
    dataset_path_labels = os.path.join(dataset_path, "labels")
    dataset_path_images = os.path.join(dataset_path, "images")

    # Create necessary directories
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(dataset_path_labels, exist_ok=True)
    os.makedirs(dataset_path_images, exist_ok=True)

    # Convert labelCloud labels to YOLO format
    labeled_indices = process_label_cloud_labels_to_yolo_format(
        config.PREPROCESSING_IMAGE_WIDTH, config.PREPROCESSING_IMAGE_HEIGHT, 
        config.PREPROCESSING_X_RANGE, config.PREPROCESSING_Y_RANGE, 
        config.PREPROCESSING_X_BINS, config.PREPROCESSING_Y_BINS, 
        config.LABEL_CLOUD_LABEL_FOLDER, config.YOLO_LABEL_FOLDER
    )
    # Transform PCD files to BEV training data
    for filename in os.listdir(config.RAW_PCD_FOLDER):
        transform_to_bev_training_data(filename, labeled_indices)
        
    # Create a new dataset
    create_new_dataset(
        config.BEV_IMAGE_FOLDER, config.YOLO_LABEL_FOLDER, 
        config.PREPROCESSING_TRAIN_SPLIT_RATIO, dataset_path
    )

    

if __name__ == "__main__":
    main()
