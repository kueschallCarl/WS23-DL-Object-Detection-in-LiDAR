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

def inference_single_image(model, image, device="cuda"):
    """
    Perform inference on a single image using a YOLO model without using DataLoader.

    Args:
        model (torch.nn.Module): The YOLO model.
        image (numpy.ndarray): A numpy array representing the input image.
        confidence_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU threshold for non-maximum suppression.
        anchors (list of tuples): Anchor boxes for YOLO.
        device (str, optional): Device to perform inference on.

    Returns:
        list: A list of bounding boxes with class predictions.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Convert image to PIL and apply transformations
    img = Image.fromarray(image).convert("RGB")
    img_transformed = config.inference_transforms(image=np.array(img))['image']
    img_transformed = torch.tensor(img_transformed).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        predictions = model(img_transformed.to(device))

    bboxes_pred = utils.get_inference_bboxes(predictions, model, iou_threshold=config.INFERENCE_IOU_THRESHOLD, anchors=config.ANCHORS, confidence_threshold=config.INFERENCE_CONFIDENCE_THRESHOLD, device=device)

    # Return to training mode if needed elsewhere
    model.train()
    return bboxes_pred