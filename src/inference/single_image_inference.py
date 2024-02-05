import torch
import torchvision.transforms as transforms
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import config
from ..utils import utils

class YOLOInferenceDatasetSingleImage(Dataset):
    """
    Custom dataset class for YOLO inference on a single image.

    Attributes:
        image_array (numpy.ndarray): A numpy array representing the input image.
        transform (callable, optional): A transformation function to apply to the image.

    Methods:
        __len__(): Returns the length of the dataset (always 1 for single image inference).
        __getitem__(index): Loads and transforms the image for inference.

    Args:
        image_array (numpy.ndarray): A numpy array representing the input image.
        transform (callable, optional): A transformation function to apply to the image.
    """
    def __init__(self, image_array, transform=None):
        self.image_array = image_array
        self.transform = transform

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset (always 1 for single image inference).
        """
        return 1  # Only one image for inference

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item to retrieve (ignored for single image inference).

        Returns:
            PIL.Image.Image: The preprocessed image for inference.
        """
        img = Image.fromarray(self.image_array).convert("RGB")

        if self.transform:
            img_np = np.array(img)
            transformed = self.transform(image=img_np)
            img = transformed["image"]

        return img

def post_process_output(dataloader, model, confidence_threshold, iou_threshold, anchors):
    """
    Post-processes model predictions to obtain bounding boxes.

    Args:
        dataloader (DataLoader): DataLoader containing the image for inference.
        model (torch.nn.Module): The YOLO model.
        confidence_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU threshold for non-maximum suppression.
        anchors (list of tuples): Anchor boxes for YOLO.

    Returns:
        list: A list of bounding boxes with class predictions.
    """
    bboxes_pred = utils.get_inference_bboxes(dataloader, model, iou_threshold, anchors, confidence_threshold, box_format="midpoint", device="cuda") 
    return bboxes_pred

def inference_single_image(model, image, confidence_threshold = config.INFERENCE_CONFIDENCE_THRESHOLD, nms_threshold = config.INFERENCE_IOU_THRESHOLD):
    """
    Perform inference on a single image using a YOLO model.

    Args:
        model (torch.nn.Module): The YOLO model.
        image (numpy.ndarray): A numpy array representing the input image.
        confidence_threshold (float, optional): Confidence threshold for filtering detections.
        nms_threshold (float, optional): IoU threshold for non-maximum suppression.

    Returns:
        list: A list of bounding boxes with class predictions.
    """
    # Load and preprocess  image
    dataset = YOLOInferenceDatasetSingleImage(image, transform=config.inference_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    # Move the model to the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Post-process the output
    bboxes_pred = post_process_output(dataloader, model, confidence_threshold, nms_threshold, anchors=config.ANCHORS)
    return bboxes_pred
