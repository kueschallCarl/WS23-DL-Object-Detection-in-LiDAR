import torch
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd
import torch

import model_training.config as config
import model_training.utils as utils

from PIL import Image
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

class YOLOInferenceDatasetSingleImage(Dataset):
    def __init__(self, image_array, transform=None):
        self.image_array = image_array
        self.transform = transform

    def __len__(self):
        return 1  # Only one image for inference

    def __getitem__(self, index):
        img = Image.fromarray(self.image_array).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

def post_process_output(dataloader, model, confidence_threshold, iou_threshold, anchors):
    bboxes_pred, bboxes_gtruth = utils.get_evaluation_bboxes(dataloader, model, iou_threshold, anchors, confidence_threshold, box_format="midpoint", device="cuda")
    return bboxes_pred[2], bboxes_pred[3], bboxes_pred[4], bboxes_pred[5], bboxes_pred[6]

def inference_single_image(model, image, confidence_threshold = config.INFERENCE_CONFIDENCE_THRESHOLD, nms_threshold = config.INFERENCE_IOU_THRESHOLD):
    # Load and preprocess the image
    dataset = YOLOInferenceDatasetSingleImage(image, transform=config.test_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    # Move the model to the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Post-process the output
    confidence_score, x, y, w, h = post_process_output(dataloader, model, confidence_threshold, nms_threshold, anchors = config.ANCHORS)
    return confidence_score, x, y, w, h


    
