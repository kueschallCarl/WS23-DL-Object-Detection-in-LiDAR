import config as config
import torch
import os
import sys
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from src.utils.logger import Tee
from src.utils.utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    save_training_results
)
from src.training.loss import YoloLoss
if config.USE_MEDIUM_MODEL:
    from src.model.model import YOLOv3_medium as YOLOv3
else:
    from src.model.model import YOLOv3
import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Train function for one epoch.

    Args:
        train_loader: DataLoader for training data.
        model: YOLOv3 model.
        optimizer: Optimizer for model parameters.
        loss_fn: YOLO loss function.
        scaler: GradScaler for mixed-precision training.
        scaled_anchors: Anchors scaled to match the model's input size.

    Returns:
        mean_loss: Mean loss over the epoch.
    """
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

    return mean_loss

def main():
    """
    Main function for training YOLO model on Pascal VOC dataset.
    """
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path="model_training_data/datasets/" + config.DATASET + "/train.csv", test_csv_path="model_training_data/datasets/" + config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    device_name = "CPU" if config.DEVICE == "cpu" else f"GPU ({torch.cuda.get_device_name(0)})"
    print(f"Using {device_name} for training.")
    training_losses = []

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{config.RUN_TITLE}_{current_datetime}"
    
    #******************************************************************************************************
    # Set up the log file
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"model_training_data/logs/{config.RUN_TITLE}_{current_datetime}.txt"
    log_file = open(log_file_path, "w")

    # Redirect stdout to console and log file
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    #******************************************************************************************************

    for epoch in range(config.NUM_EPOCHS):
        mean_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        training_losses.append(mean_loss)

        if config.SAVE_CHECKPOINTS and (epoch > 0 and epoch % config.CHECKPOINT_SAVING_INTERVAL == 0):
            save_checkpoint(model, optimizer, epoch, run_id)

        if config.SAVE_MODEL_RESULTS and epoch == config.NUM_EPOCHS-1:
            save_training_results(model, optimizer, config.NUM_EPOCHS, run_id, training_losses)

        print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % config.EVALUATION_INTERVAL == 0:
            #plot_couple_examples(model, test_loader, 0.7, 0.5, scaled_anchors, find_optimal_confidence_threshold = True, confidence_step = 0.05)
            #check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            print("Running get_evaluation_bboxes...")
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            print("Running mean_average_precision...")
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()

    sys.stdout = original_stdout
    log_file.close()

if __name__ == "__main__":
    main()
