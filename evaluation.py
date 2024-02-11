import config as config
import torch
import torch.optim as optim
from src.utils.utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    load_checkpoint,
    get_loaders,
    plot_couple_examples,
)
if config.USE_MEDIUM_MODEL:
    from src.model.model import YOLOv3_medium as YOLOv3
else:
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


    if config.EVALUATION_PLOT_RESULTS:
        plot_couple_examples(
            model,
            test_loader,
            thresh=config.EVALUATION_CONFIDENCE_THRESHOLD,
            iou_thresh=config.EVALUATION_IOU_THRESHOLD,
            anchors=scaled_anchors,
            find_optimal_confidence_threshold=config.FIND_OPTIMAL_CONFIDENCE_THRESHOLD,
            confidence_step=config.CONFIDENCE_STEP,
            desired_num_bboxes=config.DESIRED_N_BBOXES_IN_DYNAMIC_THRESHOLD
        )

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
    #print(f"Pred Bboxes: {pred_boxes}")

if __name__ == "__main__":
    main()
