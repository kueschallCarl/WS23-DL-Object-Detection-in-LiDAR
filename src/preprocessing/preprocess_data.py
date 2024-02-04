import io
import torch.optim as optim
import matplotlib.pyplot as plt
import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import config
import json

from PIL import Image



#***********************************************************************************************************
# PCD to BEV images
def crop_point_cloud(pcd_path, distance_threshold=config.PCD_CROP_DISTANCE_THRESHOLD):
    # Load pcd file
    point_cloud = o3d.io.read_point_cloud(pcd_path)

    # Convert point cloud to NumPy array
    points = np.asarray(point_cloud.points)

    # Calculate Euclidean distance from the origin
    distances = np.linalg.norm(points, axis=1)

    cropped_points = points[distances <= distance_threshold]

    return cropped_points

def generate_and_save_birds_eye_view(points, filename, dpi=200, output_folder=None):
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
    cropped_pcd = crop_point_cloud(os.path.join(config.INFERENCE_PCD_FOLDER, filename))
    bev_image = generate_and_save_birds_eye_view(cropped_pcd, filename, dpi = 200, output_folder= config.INFERENCE_TEMP_BEV_FOLDER)
    return bev_image

def transform_to_bev_training_data(filename):
    cropped_pcd = crop_point_cloud(os.path.join(config.RAW_PCD_FOLDER, filename))
    generate_and_save_birds_eye_view(cropped_pcd, filename, dpi = 200, output_folder= config.BEV_IMAGE_FOLDER)
#***********************************************************************************************************



#***********************************************************************************************************
#LabelCloud labels to YOLO labels
def labelCloud_to_YOLO(label_file, image_width, image_height, x_range, y_range, x_bins, y_bins):
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

        # Append to list in YOLO format
        yolo_labels.append((class_id, x_center_normalized, y_center_normalized, width_normalized, height_normalized))

    return yolo_labels

def process_label_cloud_labels_to_yolo_format(image_width, image_height, x_range, y_range, x_bins, y_bins,
                                               label_cloud_label_folder = config.LABEL_CLOUD_LABEL_FOLDER, yolo_label_foler = config.YOLO_LABEL_FOLDER):
    for filename in os.listdir(label_cloud_label_folder):
        if not '_classes' in filename:
            label_file = os.path.join(label_cloud_label_folder, filename)
            yolo_labels_output_folder = yolo_label_foler
            os.makedirs(yolo_labels_output_folder, exist_ok=True)
            try:
                yolo_labels = labelCloud_to_YOLO(label_file, image_width, image_height, x_range, y_range, x_bins, y_bins)
            except Exception as e:
                print(f"Error processing {label_file}: {e}")
                continue
            output_file = os.path.join(yolo_labels_output_folder, os.path.splitext(filename)[0] + ".txt")
            
            # Write YOLO labels to the output .txt file
            with open(output_file, 'w') as f:
                for label in yolo_labels:
                    f.write(' '.join(map(str, label)) + '\n')
#***********************************************************************************************************


def main():
    pass

if __name__ == "__main__":
    main()
