import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import json

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from datetime import datetime
from ..training.dataset import YOLODataset


def iou_width_height(boxes1, boxes2):
    """
    Calculate Intersection over Union (IoU) based on width and height.

    Parameters:
        boxes1 (tensor): Width and height of the first bounding boxes.
        boxes2 (tensor): Width and height of the second bounding boxes.

    Returns:
        tensor: Intersection over union of the corresponding boxes.
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculate Intersection over Union (IoU) for predicted and ground truth bounding boxes.

    Parameters:
        boxes_preds (tensor): Predicted bounding boxes (BATCH_SIZE, 4).
        boxes_labels (tensor): Ground truth bounding boxes (BATCH_SIZE, 4).
        box_format (str): "midpoint" or "corners" specifying the format of the boxes.

    Returns:
        tensor: Intersection over union for all examples.
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.

    Parameters:
        bboxes (list): List of bounding boxes [class_pred, prob_score, x1, y1, x2, y2].
        iou_threshold (float): IoU threshold for keeping predicted bounding boxes.
        threshold (float): Confidence score threshold for filtering predictions.
        box_format (str): "midpoint" or "corners" specifying the format of the boxes.

    Returns:
        list: Bounding boxes after performing NMS.
    """
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculate mean Average Precision (mAP) for object detection.

    Parameters:
        pred_boxes (list): List of predicted bounding boxes.
        true_boxes (list): List of ground truth bounding boxes.
        iou_threshold (float): IoU threshold for correct predictions.
        box_format (str): "midpoint" or "corners" specifying the format of the boxes.
        num_classes (int): Number of classes.

    Returns:
        float: mAP value across all classes given a specific IoU threshold.
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """
    Plot predicted bounding boxes on the image.

    Parameters:
        image: Image data.
        boxes: Predicted bounding boxes.
    """
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Use a single color for all boxes
    color = 'yellow'  # You can choose any color you prefer

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        """plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=f"Class {int(class_pred)}",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0},
        )"""

    plt.show()

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    """
    Get evaluation bounding boxes using a trained model.

    Parameters:
        loader (DataLoader): DataLoader for the dataset.
        model (nn.Module): Trained YOLO model.
        iou_threshold (float): IoU threshold for NMS.
        anchors (list): List of anchor boxes.
        threshold (float): Confidence score threshold for filtering predictions.
        box_format (str): "midpoint" or "corners" specifying the format of the boxes.
        device (str): Device on which to run inference (default: "cuda").

    Returns:
        list: List of predicted and ground truth bounding boxes.
    """
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def get_inference_bboxes(predictions, model, iou_threshold, anchors, confidence_threshold, device="cuda"):
    """
    Process model predictions to obtain inference bounding boxes.

    Parameters:
        predictions (torch.Tensor): Model predictions for a single image.
        model (torch.nn.Module): Trained YOLO model, not used in this function but kept for interface compatibility.
        iou_threshold (float): IoU threshold for NMS.
        anchors (list): List of anchor boxes.
        confidence_threshold (float): Confidence score threshold for filtering predictions.
        device (str): Device on which to run inference (default: "cuda").

    Returns:
        list: List of predicted bounding boxes after applying NMS.
    """
    # Ensure model is not explicitly used inside this function as predictions are directly passed
    # model.eval()  # Assuming model is already in evaluation mode outside this function

    all_pred_boxes = []

    # Assuming predictions is a list of tensors for each scale
    for i in range(len(predictions)):
        S = predictions[i].shape[2]
        anchor = torch.tensor(anchors[i], device=device) * S
        boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)

        # Flatten the list of bboxes from different scales into a single list
        bboxes = [box for sublist in boxes_scale_i for box in sublist]

        nms_boxes = non_max_suppression(
            bboxes,
            iou_threshold=iou_threshold,
            threshold=confidence_threshold
        )

        all_pred_boxes.extend(nms_boxes)

    # model.train()  # No need to toggle back to train mode here if the model state is managed outside

    return all_pred_boxes

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Convert model predictions to bounding boxes.

    Parameters:
        predictions (tensor): Model predictions.
        anchors (list): List of anchor boxes.
        S (int): Number of cells the image is divided into.
        is_preds (bool): Whether the input is predictions or true bounding boxes.

    Returns:
        list: List of converted bounding boxes.
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    """
    Calculate class accuracy for a YOLO model.

    Parameters:
        model (nn.Module): YOLO model.
        loader (DataLoader): DataLoader for the dataset.
        threshold (float): Confidence score threshold for filtering predictions.
    """
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()


def get_mean_std(loader):
    """
    Calculate the mean and standard deviation of the dataset.

    Parameters:
        loader (DataLoader): DataLoader for the dataset.

    Returns:
        tuple: Mean and standard deviation of the dataset.
    """
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, epochs, run_id):
    """
    Save a checkpoint of the model and optimizer.

    Parameters:
        model (nn.Module): YOLO model.
        optimizer (Optimizer): Optimizer used for training.
        epochs (int): Number of training epochs.
        run_id (str): Unique identifier for the run.
    """
    print("=> Saving checkpoint")
    folder_name = f"{run_id}"
    folder_path = os.path.join(config.TRAINING_CHECKPOINT_STORAGE_FOLDER, folder_name)
    print(f"Checkpoint folder created at: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    checkpoint_path = os.path.join(folder_path, f"checkpoint_{epochs}.pth.tar")
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved at: {checkpoint_path}")

def save_plot(training_losses, folder_path, plot_filename):
    """
    Save a training loss plot to a file.

    Parameters:
        training_losses (list): List of training losses.
        folder_path (str): Folder path to save the plot.
        filename (str): Filename of the plot.
    """
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig(os.path.join(folder_path, plot_filename))
    plt.close()

def save_training_results(model, optimizer, epochs, run_id, training_losses):
    """
    Save training results, model checkpoint, and configuration.

    Parameters:
        model (nn.Module): YOLO model.
        optimizer (Optimizer): Optimizer used for training.
        epochs (int): Number of training epochs.
        run_id (str): Unique identifier for the run.
        training_losses (list): List of training losses.
    """
    print("=> Saving Model Results")
    folder_name = f"{run_id}"
    folder_path = os.path.join(config.TRAINING_RESULTS_FOLDER, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    checkpoint_path = os.path.join(folder_path, f"checkpoint_{epochs}.pth.tar")
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, f"my_checkpoint.pth.tar")

    # Save configuration values in a JSON file
    config_values = {
        "NUM_EPOCHS": config.NUM_EPOCHS,
        "BATCH_SIZE": config.BATCH_SIZE,
        "LEARNING_RATE": config.LEARNING_RATE,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,
        "LOAD_MODEL": config.LOAD_MODEL,
        "DATASET": config.DATASET,
        "NUM_CLASSES": config.NUM_CLASSES,
    }

    json_filename = f"config_values_{run_id}.json"
    json_path = os.path.join(config.TRAINING_RESULTS_FOLDER, folder_name, json_filename)

    with open(json_path, 'w') as json_file:
        json.dump(config_values, json_file, indent=4)

    plot_filename = f"{run_id}_learning_curve_loss.png"
    save_plot(training_losses, os.path.join(config.TRAINING_RESULTS_FOLDER, folder_name), plot_filename)

    print(f"Model Results saved at: {checkpoint_path} and configuration saved at: {json_path}")


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load a model checkpoint and optimizer state.

    Parameters:
        checkpoint_file (str): Path to the checkpoint file.
        model (nn.Module): YOLO model.
        optimizer (Optimizer): Optimizer used for training.
        lr (float): Learning rate for the optimizer.
    """
    print(f"=> Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path):
    """
    Get DataLoader instances for training and testing.

    Parameters:
        train_csv_path (str): Path to the training CSV file.
        test_csv_path (str): Path to the testing CSV file.

    Returns:
        DataLoader: DataLoader for training dataset.
        DataLoader: DataLoader for testing dataset.
        DataLoader: DataLoader for evaluation during training.
    """
    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers= True,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers= True,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers= True,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def dynamic_threshold(model, loader, iou_thresh, anchors, confidence_step, desired_num_bboxes=config.DESIRED_N_BBOXES_IN_DYNAMIC_THRESHOLD):
    """
    Calculate a dynamic confidence threshold for object detection.

    Parameters:
        model (nn.Module): YOLO model.
        loader (DataLoader): DataLoader for the dataset.
        iou_thresh (float): IoU threshold for NMS.
        anchors (list): List of anchor boxes.
        confidence_step (float): Confidence step for threshold adjustment.
        desired_num_bboxes (int): Desired number of bounding boxes.

    Returns:
        float: Dynamic confidence threshold.
    """
    print("Starting dynamic thresholding")
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        model.train()

    confidence_threshold = 1.0
    num_bboxes = 0
    while num_bboxes < desired_num_bboxes:
        confidence_threshold -= confidence_step  # Adjust this increment based on your needs
        nms_boxes = non_max_suppression(
            bboxes[0], iou_threshold=iou_thresh, threshold=confidence_threshold, box_format="midpoint",
        )
        num_bboxes = len(nms_boxes)
        print(f"Current Thresh: {confidence_threshold}")
        print(f"Current num Boxes: {num_bboxes}")

    print(f"Confidence Threshold: {confidence_threshold}")
    return confidence_threshold


def plot_couple_examples(model, loader, thresh, iou_thresh, anchors, find_optimal_confidence_threshold=False, confidence_step = 0.05):
    """
    Plot examples with bounding boxes using a trained YOLO model.

    Parameters:
        model (nn.Module): Trained YOLO model.
        loader (DataLoader): DataLoader for the dataset.
        thresh (float): Confidence score threshold for filtering predictions.
        iou_thresh (float): IoU threshold for NMS.
        anchors (list): List of anchor boxes.
        find_optimal_confidence_threshold (bool): Whether to dynamically find the confidence threshold.
        confidence_step (float): Confidence step for threshold adjustment.
    """
    print("Starting plotting of examples")
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    if find_optimal_confidence_threshold:
        threshold = dynamic_threshold(model, loader, iou_thresh, anchors, confidence_step)
    else:
        threshold = thresh  # Set a default threshold if not using dynamic thresholding

    for i in range(batch_size):
        print(f"Bboxes at {i}: {len(bboxes[i])}")
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=threshold, box_format="midpoint",
        )
        print(f"nms_boxes at {i}: {len(nms_boxes)} over confidence: {threshold} and under IOU threshold: {iou_thresh}")

        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)


def seed_everything(seed=42):
    """
    Seed all random number generators for reproducibility.

    This function sets the seed for various random number generators, including Python's random,
    NumPy's random, and PyTorch's random number generators. It also ensures deterministic behavior
    for CUDA operations using torch.backends.cudnn.

    Parameters:
        seed (int): Seed value for random number generators (default: 42).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python's hash seed
    random.seed(seed)  # Seed Python's random module
    np.random.seed(seed)  # Seed NumPy's random module
    torch.manual_seed(seed)  # Seed PyTorch's random module for CPU
    torch.cuda.manual_seed(seed)  # Seed PyTorch's random module for a single GPU
    torch.cuda.manual_seed_all(seed)  # Seed PyTorch's random module for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CUDA operations
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmark mode for deterministic results