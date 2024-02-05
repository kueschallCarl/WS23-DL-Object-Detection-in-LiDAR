import config as config
import torch
import torch.optim as optim
from src.utils.utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
)
from src.model.model import YOLOv3

def main():
    # Initialize YOLOv3 model
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Initialize Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Load data loaders
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path="model_training_data/datasets/" + config.DATASET + "/train.csv",
        test_csv_path="model_training_data/datasets/" + config.DATASET + "/test.csv"
    )

    # Load checkpoint if specified
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE,
            model,
            optimizer,
            config.LEARNING_RATE
        )

    # Calculate scaled anchors
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # Determine device for plotting
    device_name = "CPU" if config.DEVICE == "cpu" else f"GPU ({torch.cuda.get_device_name(0)})"
    print(f"Using {device_name} for plotting.")

    # Plot example results (comment this out if you wish to skip the visualization)
    """plot_couple_examples(
        model,
        test_loader,
        iou_thresh=0.7,
        confidence_thresh=0.2,
        anchors=scaled_anchors,
        find_optimal_confidence_threshold=True,
        confidence_step=0.01
    )"""

    print("Running get_evaluation_bboxes...")
    # Get evaluation bounding boxes
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD
    )

    print("Running mean_average_precision...")
    # Calculate mean average precision
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES
    )

    # Print mean average precision
    print(f"MAP: {mapval.item()}")
    print(f"Pred Bboxes: {pred_boxes}")

if __name__ == "__main__":
    main()
