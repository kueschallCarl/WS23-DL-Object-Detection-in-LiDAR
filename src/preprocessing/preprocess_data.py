import io
import torch.optim as optim
import matplotlib.pyplot as plt
import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import config
import random
import csv
import json
import open3d as o3d

from PIL import Image

#***********************************************************************************************************
# PCD to BEV images
def crop_point_cloud(pcd_path, distance_threshold=config.PCD_CROP_DISTANCE_THRESHOLD):
    """
    Crop a 3D point cloud based on a specified distance threshold from the origin.

    Args:
        pcd_path (str): Path to the input point cloud file (in PCD format).
        distance_threshold (float): Distance threshold for cropping points (default: config.PCD_CROP_DISTANCE_THRESHOLD).

    Returns:
        np.ndarray: Cropped point cloud as a NumPy array.
    """
    # Load pcd file
    point_cloud = o3d.io.read_point_cloud(pcd_path)

    # Convert point cloud to NumPy array
    points = np.asarray(point_cloud.points)

    # Calculate Euclidean distance from the origin
    distances = np.linalg.norm(points, axis=1)

    cropped_points = points[distances <= distance_threshold]

    return cropped_points

def save_point_cloud_as_pcd(point_cloud, output_pcd_path):
    """
    Save a point cloud as a .pcd file.

    Args:
        point_cloud (np.ndarray): The input point cloud as a NumPy array.
        output_pcd_path (str): The path to save the .pcd file.
    """
    # Convert the NumPy array to a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Check if the folder specified in output_pcd_path exists, and create it if necessary
    output_folder = os.path.dirname(output_pcd_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the PointCloud object as a .pcd file
    o3d.io.write_point_cloud(output_pcd_path, pcd)

def generate_and_save_birds_eye_view(points, filename, dpi=200, output_folder=None):
    """
    Generate and save a Bird's Eye View (BEV) image from a cropped 3D point cloud.

    Args:
        points (np.ndarray): Cropped point cloud as a NumPy array.
        filename (str): Name of the output image file.
        dpi (int): Dots per inch for the output image (default: 200).
        output_folder (str): Folder to save the output image (default: None).

    Returns:
        np.ndarray: The generated BEV image as a NumPy array.
    """
    x_bins = np.linspace(-5, 5, 250)
    y_bins = np.linspace(-5, 5, 250)
    x_range = [-5, 5]
    y_range = [-5, 5]

    # Filter points
    mask = ((points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) & 
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]))
    points = points[mask]
    
    # Compute histogram
    hist, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=(x_bins, y_bins))
    hist_normalized = np.log(hist + 1)
    
    # Generate a plot
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    ax.imshow(hist_normalized.T, cmap='viridis', origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    ax.axis('off')
    
    # Save the image to disk if output folder is provided
    if output_folder:
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Save the image to disk
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Image saved at {output_path}")
        plt.close(fig)  # Close the plot if saved
    else:
        print("Output folder not provided, image not saved.")
        output_path = None  # Set output_path to None if not saving
    
    # Load the saved image and return
    if output_path:
        saved_image = Image.open(output_path).convert("RGB")
        return np.array(saved_image)
    else:
        print("Output folder not provided, image not saved.")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig) 
        return np.array(img)

def transform_to_bev_inference(filename):
    """
    Transform a point cloud file to a Bird's Eye View (BEV) image for inference.

    Args:
        filename (str): Name of the input point cloud file.

    Returns:
        np.ndarray: The generated BEV image as a NumPy array.
    """
    cropped_pcd = crop_point_cloud(os.path.join(config.INFERENCE_PCD_FOLDER, filename))
    bev_image = generate_and_save_birds_eye_view(cropped_pcd, filename, dpi=200, output_folder=config.INFERENCE_TEMP_BEV_FOLDER)
    return bev_image

def transform_to_bev_training_data(filename, labeled_indices):
    """
    Transform a point cloud file to a Bird's Eye View (BEV) image for training data.

    Args:
        filename (str): Name of the input point cloud file.
    """
    if filename.split('.')[0] in labeled_indices:
        cropped_pcd = crop_point_cloud(os.path.join(config.RAW_PCD_FOLDER, filename))
        if config.STORE_CROPPED_PCD_FOR_LABELING:
            os.makedirs(config.CROPPED_PCD_FOLDER, exist_ok=True)    
            cropped_pcd_path = os.path.join(config.CROPPED_PCD_FOLDER, filename)
            save_point_cloud_as_pcd(cropped_pcd, cropped_pcd_path)
        generate_and_save_birds_eye_view(cropped_pcd, filename, dpi=200, output_folder=config.BEV_IMAGE_FOLDER)
    else:
        pass
#***********************************************************************************************************

#***********************************************************************************************************
# LabelCloud labels to YOLO labels
def labelCloud_to_YOLO(label_file, image_width, image_height, x_range, y_range, x_bins, y_bins):
    """
    Convert LabelCloud labels to YOLO format labels.

    Args:
        label_file (str): Path to the JSON label file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        x_range (list): Range of X-axis values.
        y_range (list): Range of Y-axis values.
        x_bins (int): Number of X-axis bins.
        y_bins (int): Number of Y-axis bins.

    Returns:
        tuple: A tuple containing YOLO-formatted labels as a list of tuples and a list of erroneous filenames.
    """
    # Read the JSON label file
    with open(label_file, 'r') as f:
        labels = json.load(f)

    # Calculate the size of each bin
    x_bin_size = (x_range[1] - x_range[0]) / x_bins
    y_bin_size = (y_range[1] - y_range[0]) / y_bins

    # Initialize an empty list to hold YOLO-formatted labels
    yolo_labels = []

    # Loop over each object in the label file
    for obj in labels['objects']:
        # Get the centroid coordinates
        centroid_x = obj['centroid']['x']
        centroid_y = -obj['centroid']['y']

        # Calculate the pixel position of the centroid
        x_pixel = int((centroid_x - x_range[0]) / x_bin_size)
        y_pixel = int((centroid_y - y_range[0]) / y_bin_size)

        # Convert dimensions to pixels
        width_pixel = int(obj['dimensions']['length'] / x_bin_size)
        height_pixel = int(obj['dimensions']['width'] / y_bin_size)

        # Normalize the values by the image size
        class_id = 0
        x_center_normalized = (x_pixel / image_width)
        y_center_normalized = (y_pixel / image_height)
        width_normalized = width_pixel / image_width
        height_normalized = height_pixel / image_height

        # Check if any value is outside the [0, 1] range
        if (
            0 <= x_center_normalized <= 1 and
            0 <= y_center_normalized <= 1 and
            0 <= width_normalized <= 1 and
            0 <= height_normalized <= 1
        ):
            # Append to list in YOLO format
            yolo_labels.append((class_id, x_center_normalized, y_center_normalized, width_normalized, height_normalized))
        else:
            continue

    return yolo_labels

def process_label_cloud_labels_to_yolo_format(image_width, image_height, x_range, y_range, x_bins, y_bins,
                                               label_cloud_label_folder=config.LABEL_CLOUD_LABEL_FOLDER,
                                               yolo_label_foler=config.YOLO_LABEL_FOLDER):
    """
    Process LabelCloud labels and convert them to YOLO format labels, saving them to output files.

    Args:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        x_range (list): Range of X-axis values.
        y_range (list): Range of Y-axis values.
        x_bins (int): Number of X-axis bins.
        y_bins (int): Number of Y-axis bins.
        label_cloud_label_folder (str): Folder containing LabelCloud label files (default: config.LABEL_CLOUD_LABEL_FOLDER).
        yolo_label_foler (str): Folder to save YOLO format label files (default: config.YOLO_LABEL_FOLDER).
    """
    labeled_indices = []
    for filename in os.listdir(label_cloud_label_folder):
        
        idx = filename.split('.')[0]
        if not '_classes' in filename:
            label_file = os.path.join(label_cloud_label_folder, filename)
            yolo_labels_output_folder = yolo_label_foler
            os.makedirs(yolo_labels_output_folder, exist_ok=True)
            try:
                yolo_labels = labelCloud_to_YOLO(label_file, image_width, image_height, x_range, y_range, x_bins, y_bins)
                labeled_indices.append(idx)
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
                continue
            output_file = os.path.join(yolo_labels_output_folder, os.path.splitext(filename)[0] + ".txt")
            
            # Write YOLO labels to the output .txt file
            with open(output_file, 'w') as f:
                for label in yolo_labels:
                    f.write(' '.join(map(str, label)) + '\n')
    print(f"Labeled Indices: {labeled_indices}")
    return labeled_indices
#***********************************************************************************************************
# Creating new dataset from YOLO labels and BEV Images
def create_new_dataset(image_folder, label_folder, split_ratio, csv_dir):
    """
    Create a new dataset by splitting images and labels into training and test sets, and save the split information as CSV files.

    Args:
        image_folder (str): Folder containing image files.
        label_folder (str): Folder containing label files.
        split_ratio (float): Ratio of data to split into training set (e.g., 0.8 for 80% training).
        csv_dir (str): Directory to save the CSV files.
    """
    # Get the list of image and label filenames
    image_filenames = os.listdir(image_folder)
    label_filenames = os.listdir(label_folder)
    
    # Create a dictionary to map image base names to label files
    label_file_map = {}
    for label_file in label_filenames:
        base_name = label_file.split('.')[0]  # Assumes the common part is before the first dot
        label_file_map[base_name] = label_file

    print(f"image_filenames: {image_filenames[:5]}")
    print(f"label_filenames: {label_filenames[:5]}")

    # Shuffle the image filenames
    random.shuffle(image_filenames)
    
    # Calculate the split index
    split_index = int(len(image_filenames) * split_ratio)
    
    # Split the image filenames into train and test sets
    train_image_filenames = image_filenames[:split_index]
    test_image_filenames = image_filenames[split_index:]

    print(f"image_filenames after split: {train_image_filenames[:5]}")
    print(f"test_image_filenames after split: {test_image_filenames[:5]}")
    # Function to find the corresponding label file
    def find_label_file(image_file):
        base_name = image_file.split('.')[0]  # Assumes the common part is before the first dot
        return label_file_map.get(base_name, None)

    # Create the train.csv file
    train_csv_path = os.path.join(csv_dir, 'train.csv')
    with open(train_csv_path, 'w', newline='') as train_file:
        writer = csv.writer(train_file)
        for image_filename in train_image_filenames:
            label_filename = find_label_file(image_filename)
            if label_filename:
                print(f"train.csv: label filename: {label_filename} found, now writing")
                writer.writerow([image_filename, label_filename])
    
    # Create the test.csv file
    test_csv_path = os.path.join(csv_dir, 'test.csv')
    with open(test_csv_path, 'w', newline='') as test_file:
        writer = csv.writer(test_file)
        for image_filename in test_image_filenames:
            label_filename = find_label_file(image_filename)
            if label_filename:
                print(f"test.csv: label filename: {label_filename} found, now writing")
                writer.writerow([image_filename, label_filename])

#***********************************************************************************************************

def main():
    pass

if __name__ == "__main__":
    main()
