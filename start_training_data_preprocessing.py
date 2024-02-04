"""
Use this script to transform PCD files to BEV images and labelCloud labels to YOLO format labels
"""

import config as config
import torch
import os
import sys
import torch.optim as optim
import matplotlib.pyplot as plt

from src.model.model import YOLOv3
from tqdm import tqdm
from datetime import datetime
from src.utils.logger import Tee
from src.training.loss import YoloLoss
from src.preprocessing import process_label_cloud_labels_to_yolo_format, transform_to_bev

def main():
    process_label_cloud_labels_to_yolo_format(config.PREPROCESSING_IMAGE_WIDTH, config.PREPROCESSING_IMAGE_HEIGHT, 
                                              config.PREPROCESSING_X_RANGE, config.PREPROCESSING_Y_RANGE, config.PREPROCESSING_X_BINS,
                                              config.PREPROCESSING_Y_BINS, config.LABEL_CLOUD_LABEL_FOLDER, config.YOLO_LABEL_FOLDER)
    for filename in os.listdir(config.RAW_PCD_FOLDER):
        transform_to_bev(filename)

if __name__ == "__main__":
    main()
