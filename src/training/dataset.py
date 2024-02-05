"""
Creates a Pytorch dataset to load the Pascal VOC
"""

# import necessary libraries
import numpy as np
import os
import pandas as pd
import torch
import config

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from ..utils import utils

# Set a flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=1,
        transform=None,
    ):
        """
        Initializes the YOLO dataset.

        Args:
            csv_file (str): Path to the CSV file containing dataset annotations.
            img_dir (str): Directory containing image files.
            label_dir (str): Directory containing label files.
            anchors (list): List of anchor sizes for YOLO.
            image_size (int): Size of the input images (default is 416).
            S (list): List of scales for YOLO predictions (default is [13, 26, 52]).
            C (int): Number of classes (default is 20).
            transform (callable): Optional data transformation to apply to images and labels.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and YOLO targets.
        """
        # Get the label file path and load bounding boxes
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()

        # Get the image file path and load the image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply data augmentation if specified
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Initialize targets for YOLO predictions
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        # Process each bounding box
        for box in bboxes:
            # Calculate IoU with anchor boxes
            iou_anchors = utils.iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            
            # Iterate through anchor indices and assign targets
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # Determine the cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                if not anchor_taken and not has_anchor[scale_idx]:
                    # Assign target values for the selected anchor
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # Calculate position within cell
                    width_cell, height_cell = width * S, height * S  # Relative to cell
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Ignore prediction for this anchor if IoU is above the threshold
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)