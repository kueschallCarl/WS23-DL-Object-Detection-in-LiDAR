import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from src.utils.utils import seed_everything

#***********************************************************************************************************
#Training Settings
#use 'start_run_training.py' to start this process
DATASET = 'BENCHMARK_DATASET_400' #The name of the dataset in the dataset folder that should be used for training and evaluation.
RUN_TITLE = 'BENCHMARK_RUN_400' #This title is a convenience and logging measure, to easily identify output files and directories.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #use GPU computing if possible.
seed_everything()  # If you want deterministic behavior.
NUM_WORKERS = 4 #The number of workers the Data Loader should use.
BATCH_SIZE = 32 #The amount of samples that are input to the model at a time.
IMAGE_SIZE = 416 #The image size.
NUM_CLASSES = 1 #The total amount of classes in the Dataset (1 for cone detection).
LEARNING_RATE = 1e-5 #The rate at which the model performs backpropagation (adjusts parameters after forward pass) (Very high learning rates can lead to overshooting, very low learning rates result in a lengthy training run, or in the worst case the model not learning at all).
WEIGHT_DECAY = 1e-4 #The rate at which the learning rate decays during training (As the model improves, a lower learning rate is required to carefully extract the last bit of potential performance).
NUM_EPOCHS = 50 #The number of epochs, the model should be trained for.

USE_MEDIUM_MODEL = True #Set True if you wish to use the smaller YOLOv3 model, with approx. 8mil params compared to the 60+mil params of the original.
PLOT_EXAMPLES_DURING_TRAINING = False #set true if you wish to plot examples during training at the interval of EVALUATION_INTERVAL

CONF_THRESHOLD = 0.7 #In evaluation and during inference, only bounding boxes (bboxes) that the model predicted with a confidence higher than this threshold will be kept.
MAP_IOU_THRESH = 0.2 #The intersection over union threshold to be used when computing the MAP (Mean Average Precision) metric.
NMS_IOU_THRESH = 0.2 #The intersection over union threshold to be used when running Non-Maximum-Suppression. (Filtering out low-confidence bboxes that overlap with others more than this threshold)
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8] #The three Scales (Resolutions) at which YOLOv3 attempts to find bboxes.

PIN_MEMORY = True #Enables faster data transfer from the CPU to the GPU
LOAD_MODEL = True #Will load a model from the checkpoint file specified at CHECKPOINT_FILE. Always set true for evaluation and inference. Set false for a fresh training run. Set true to fine-tune a model, training it further.
SAVE_MODEL_RESULTS = True #Set true if you wish to store training metadata, plots, and the last model checkpoint after training.
SAVE_CHECKPOINTS = True #Set true if you wish to store checkpoints at the interval specified at CHECKPOINT_SAVING_INTERVAL.
CHECKPOINT_SAVING_INTERVAL = 20 #The interval at which checkpoints should be stored at the location specified at CHECKPOINT_FILE.
EVALUATION_INTERVAL = 20 #The interval at which the script should run evaluation during training (Will print the MAE etc.)

CHECKPOINT_FILE = 'model_inference_data/model/benchmark_400_medium.pth.tar' #The file from which a model should be loaded!!! Use INFERENCE_CHECKPOINT_FILE to define which checkpoint to load at inference time.
IMG_DIR = "model_training_data/datasets/" + DATASET + "/images/" #The directory containing BEV images of the dataset specified at DATASET.
LABEL_DIR = "model_training_data/datasets/" + DATASET + "/labels/" #The directory containing YOLO labels of the dataset specified at DATASET.
TRAINING_RESULTS_FOLDER = "model_training_data/model_results/" #The folder in which training results will be stored, should they be enabled at SAVE_MODEL_RESULTS.
TRAINING_CHECKPOINT_STORAGE_FOLDER = "model_training_data/checkpoint_storage/" #The folder in which training checkpoints will be stored at the interval specified at CHECKPOINT_SAVING_INTERVAL.

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1] The anchors essentially provide bbox starting points for the model.
#***********************************************************************************************************

#***********************************************************************************************************
#Evaluation Settings
#use 'evaluation.py' to start this process
EVALUATION_PLOT_RESULTS = True #Set True if you wish to plot results for the test set

FIND_OPTIMAL_CONFIDENCE_THRESHOLD = False #Set True if you wish to dynamically find a threshold that will result in DESIRED_N_BBOXES_IN_DYNAMIC_THRESHOLD BBoxes in the visualization and MAP calculation
DESIRED_N_BBOXES_IN_DYNAMIC_THRESHOLD = 8 #The amount of bboxes that the dynamic thresholding should result in
CONFIDENCE_STEP = 0.05 #The step the dynamic thresholding takes in each iteration

EVALUATION_CONFIDENCE_THRESHOLD = 0.65 #The confidence threshold in evaluation. Use this when not running dynamic thresholding
EVALUATION_IOU_THRESHOLD = 0.2 #The IOU (overlapping) threshold in evaluation. THIS IS INDEPENDENT OF DYNAMIC THRESHOLDING
#***********************************************************************************************************

#***********************************************************************************************************
#Preprocessing Settings
#use 'start_training_data_preprocessing.py' to start this process
PCD_CROP_DISTANCE_THRESHOLD = 5.0 #The maximum distance at which point-cloud points will be kept during cropping in preprocessing.

NEW_DATASET_NAME = 'BENCHMARK_DATASET_400' #The name given to the new Dataset, that will be created in preprocessing.

RAW_PCD_FOLDER = 'label_cloud_project/datastore/pointclouds/raw_pcds' #The folder containing the '.pcd' files for preprocessing.
LABEL_CLOUD_LABEL_FOLDER = 'label_cloud_project/datastore/labels/label_cloud_labels' #The folder containing the labelCloud labels for preprocessing.

YOLO_LABEL_FOLDER = f'model_training_data/datasets/{NEW_DATASET_NAME}/labels' #The destination folder for YOLO labels
BEV_IMAGE_FOLDER = f'model_training_data/datasets/{NEW_DATASET_NAME}/images' #The destination folder for the BEV images
CROPPED_PCD_FOLDER = f'model_training_data/datasets/{NEW_DATASET_NAME}/pcd' #The destination folder for cropped PCD files

STORE_CROPPED_PCD_FOR_LABELING = False #ENABLE THIS TO RECEIVE THE CROPPED PCDS NECESSARY FOR LABELING IN labelCloud!!! Not necessary in any other scenario.

PREPROCESSING_TRAIN_SPLIT_RATIO = 0.9 #The split ratio n-test-samples / n-train-samples. A split of 0.7 would mean that 70% of all samples will be used for training and 30% for testing.

#These settings configure the process that transforms point-clouds to BEV images
PREPROCESSING_IMAGE_WIDTH = 250
PREPROCESSING_IMAGE_HEIGHT = 250
PREPROCESSING_X_RANGE = [-5, 5]
PREPROCESSING_Y_RANGE = [-5, 5]
PREPROCESSING_X_BINS = 250
PREPROCESSING_Y_BINS = 250
#***********************************************************************************************************


#***********************************************************************************************************
#Inference Settings
#use 'start_run_inference.py' to start this process
INFERENCE_RUN_TITLE = 'BENCHMARK_400_INFERENCE_MEDIUM' #This title is a convenience and logging measure, to easily identify output files and directories. Just for inference runs this time.

INFERENCE_CHECKPOINT_FILE = 'model_inference_data/model/benchmark_400_medium.pth.tar' #The checkpoint file to use for inference.

INFERENCE_PCD_FOLDER = 'model_inference_data/pcd/' #The folder in which the inference script will wait for '.pcd' files to arrive.
INFERENCE_TEMP_BEV_FOLDER = '' #SET TO EMPTY STRING FOR INFERENCE! COSTS TIME! A folder to store the BEV images that result from inference preprocessing (mainly for dev purposes)
INFERENCE_RESULTS_FOLDER = 'model_inference_data/inference_results/' #The folder in which inference results are logged (contains Folders named by the INFERENCE_RUN_TITLE, which contain the json file with all predictions)
INFERENCE_PROCESSED_PCD_FOLDER = 'model_inference_data/processed_pcds/' #A folder in which processed '.pcd' files will be moved, to avoid processing the same file more than once.

INFERENCE_SHOW_LIVE_RESULTS = False #Set to true if you wish to visualize the inference output as it arrives, live.
INFERENCE_CONFIDENCE_THRESHOLD = 0.6 #Only predictions with confidence greater than this threshold will be kept.
INFERENCE_IOU_THRESHOLD = 0.2 #Bboxes that overlap with a bbox, that has a higher confidence to a degree greater than this threshold, will be deleted.
#***********************************************************************************************************


#***********************************************************************************************************
#Albumentations 
#All transformations and data augmentation procedures applied to input data are listed here.
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


#***********************************************************************************************************
#Classes
BEV_BATCH1_CLASSES = ["cone"]
#***********************************************************************************************************
