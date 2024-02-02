import config as config
import torch
import torch.optim as optim
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from model import YOLOv3

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )  # Note: Passing None for optimizer and learning rate since they are not needed for plotting

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    device_name = "CPU" if config.DEVICE == "cpu" else f"GPU ({torch.cuda.get_device_name(0)})"
    print(f"Using {device_name} for plotting.")

    plot_couple_examples(model, test_loader, 0.4, 0.2, scaled_anchors)

if __name__ == "__main__":
    main()
