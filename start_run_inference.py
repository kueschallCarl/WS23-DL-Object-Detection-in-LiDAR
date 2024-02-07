import json
import os
import numpy as np
import time
import threading
import torch
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk
from mpl_toolkits.mplot3d import Axes3D
import config
import src.utils.utils as utils
import src.inference.single_image_inference as inference
import src.preprocessing.preprocess_data as preprocessing
from model.model_medium import YOLOv3
from src.inference.plot_results import plot_bounding_boxes, plot_bounding_box_single_image, ImageDisplayApp

global_app = None

def run_tkinter():
    global global_app
    root = tk.Tk()
    global_app = ImageDisplayApp(root)
    root.mainloop()

def main():
    # Generate a unique run ID using the current datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{config.INFERENCE_RUN_TITLE}_{current_datetime}"

    # Create folders for results and processed point clouds
    results_folder = os.path.join(config.INFERENCE_RESULTS_FOLDER, run_id)
    processed_pcd_folder = config.INFERENCE_PROCESSED_PCD_FOLDER
    results_file_path = os.path.join(results_folder, f"results_{run_id}.json")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(processed_pcd_folder, exist_ok=True)

    if config.INFERENCE_SHOW_LIVE_RESULTS:
        tkinter_thread = threading.Thread(target=run_tkinter)
        tkinter_thread.start()

    # Initialize YOLOv3 model and load checkpoint
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    utils.load_checkpoint(config.INFERENCE_CHECKPOINT_FILE, model, None, None)

    # Inference loop
    print(f"Waiting for pcd files in {config.RAW_PCD_FOLDER}")

    """
    #Use this for profiling
    flag = 0
    while True and flag < 25:
"""
    while True:
        # Monitor the folder for new point cloud files
        for file_name in os.listdir(config.INFERENCE_PCD_FOLDER)[:3]:
            print(file_name)
            if file_name.endswith(".pcd"):
                # Transform point cloud to BEV image
                bev_image = preprocessing.transform_to_bev_inference(file_name)

                # Extract x, y coordinates from YOLOv3 output
                bboxes_pred = inference.inference_single_image(model, bev_image)

                # Print results to console
                print(f"File: {file_name}, bboxes: {bboxes_pred}")

                result_dict = {}
                for idx in range(len(bboxes_pred)):
                    bbox_info = {
                        "confidence": bboxes_pred[idx][1],
                        "x": bboxes_pred[idx][2],
                        "y": bboxes_pred[idx][3],
                        "w": bboxes_pred[idx][4],
                        "h": bboxes_pred[idx][5]
                    }
                    result_dict[idx] = bbox_info

                # Create a dictionary to store all results
                results = {}

                # Check if the results file already exists
                if os.path.exists(results_file_path):
                    # Read existing results from the file
                    with open(results_file_path, 'r') as json_file:
                        results = json.load(json_file)

                # Add new results to the dictionary
                results[file_name] = result_dict

                #Visualize the models' performance in real-time, by plotting the predicted bboxes on top of the corresponding BEV image
                if config.INFERENCE_SHOW_LIVE_RESULTS:
                    plot_bounding_box_single_image(result_dict, config.INFERENCE_TEMP_BEV_FOLDER, file_name, global_app)

                # Write all results back to the file
                with open(results_file_path, "w") as json_file:
                    json.dump(results, json_file, indent=2)

                # Optional: Move processed file to another folder to avoid re-processing
                os.rename(os.path.join(config.INFERENCE_PCD_FOLDER, file_name),
                          os.path.join(processed_pcd_folder, file_name))

            # Sleep for a while before checking for new files again
            time.sleep(0.01)
            #flag +=1
            #print(f"flag after update: {flag}")
    

if __name__ == "__main__":
    main()
