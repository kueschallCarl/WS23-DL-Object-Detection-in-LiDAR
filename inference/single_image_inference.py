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


from model_training.utils import (
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
            img_np = np.array(img)
            transformed = self.transform(image=img_np)
            img = transformed["image"]

        return img

def post_process_output(dataloader, model, confidence_threshold, iou_threshold, anchors):
    bboxes_pred = utils.get_inference_bboxes(dataloader, model, iou_threshold, anchors, confidence_threshold, box_format="midpoint", device="cuda") 
    return bboxes_pred

def inference_single_image(model, image, confidence_threshold = config.INFERENCE_CONFIDENCE_THRESHOLD, nms_threshold = config.INFERENCE_IOU_THRESHOLD):
    # Load and preprocess  image
    dataset = YOLOInferenceDatasetSingleImage(image, transform=config.inference_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory = config.PIN_MEMORY)

    # Move the model to the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Post-process the output
    bboxes_pred = post_process_output(dataloader, model, confidence_threshold, nms_threshold, anchors = config.ANCHORS)
    return bboxes_pred


    
