"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""

import random
import torch
import torch.nn as nn

from ..utils.utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss (Weights)
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        """
        Forward pass of the YoloLoss function.

        Args:
            predictions (torch.Tensor): Predictions from the YOLO model.
            target (torch.Tensor): Target values for the predictions.
            anchors (torch.Tensor): Anchor boxes used for predictions.

        Returns:
            torch.Tensor: Calculated Yolo loss.
        """

        # Check where obj and noobj (we ignore if target == -1)
        #Checks for grid cells in which an object is present (confidence == 1)
        obj = target[..., 0] == 1  # in the paper this is Iobj_i
        #Checks for grid cells in which an object is not present (confidence == 0)
        noobj = target[..., 0] == 0  # in the paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        #This loss encourages the model to predict low confidence scores in grid cells where there are no objects.
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        #This loss encourages the model to predict high confidence scores when an object is present and align the predicted bounding boxes with the ground truth
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        #This loss helps in refining the predicted bounding boxes to match the ground truth
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x, y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        """
        In YOLO, after the confidence score and bounding box coordinates, the remaining values represent class probabilities for different object classes.
        [..., 5:] selects all elements in the last dimension starting from index 5, effectively extracting the predicted class probabilities
        """
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        #print("__________________________________")
        #print(self.lambda_box * box_loss)
        #print(self.lambda_obj * object_loss)
        #print(self.lambda_noobj * no_object_loss)
        #print(self.lambda_class * class_loss)
        #print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
