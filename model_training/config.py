import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from .utils import seed_everything


#***********************************************************************************************************
#Training Settings
DATASET = 'BEV_BATCH1'
RUN_TITLE = 'First_Tests'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
CONF_THRESHOLD = 0.7
MAP_IOU_THRESH = 0.2
NMS_IOU_THRESH = 0.2
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL_RESULTS = True
SAVE_CHECKPOINTS = True
CHECKPOINT_FILE = "my_checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]
#***********************************************************************************************************


#***********************************************************************************************************
#Preprocessing Settings
PCD_CROP_DISTANCE_THRESHOLD = 5.0

RAW_PCD_FOLDER = 'label_cloud_project/datastore/pointclouds/raw_pcds'
CROPPED_PCD_FOLDER = 'label_cloud_project/datastore/pointclouds/cropped_pcds'  
LABEL_CLOUD_LABEL_FOLDER = 'label_cloud_project/datastore/labels/label_cloud_labels'
YOLO_LABEL_FOLDER = 'label_cloud_project/datastore/labels/yolo_labels'  
BEV_IMAGE_FOLDER = 'label_cloud_project/datastore/images/birds_eye_view_images'
#***********************************************************************************************************


#***********************************************************************************************************
#Inference Settings
INFERENCE_RUN_TITLE = 'First_Test_Inference'

INFERENCE_CHECKPOINT_FILE = 'inference/model/inference_model_checkpoint.pth.tar'

INFERENCE_PCD_FOLDER = 'inference/pcd/'
INFERENCE_TEMP_BEV_FOLDER = 'inference/temp/bev_images'
INFERENCE_RESULTS_FOLDER = 'inference/inference_results/'
INFERENCE_PROCESSED_PCD_FOLDER = 'inference/processed_pcds/'

INFERENCE_STORE_BEV_IMAGES = True
INFERENCE_CONFIDENCE_THRESHOLD = 0.7
INFERENCE_IOU_THRESHOLD = 0.2
#***********************************************************************************************************

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
inference_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
)

#***********************************************************************************************************
#Classes
"""
PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]"""

BEV_BATCH1_CLASSES = ["cone"]
#***********************************************************************************************************
