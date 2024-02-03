import io
import torch.optim as optim
import matplotlib.pyplot as plt
import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

import model_training.config as config

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
    if output_folder and config.INFERENCE_STORE_BEV_IMAGES:
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

def transform_to_bev(filename):
    cropped_pcd = crop_point_cloud(os.path.join(config.INFERENCE_PCD_FOLDER, filename))
    bev_image = generate_and_save_birds_eye_view(cropped_pcd, filename, dpi = 200, output_folder= config.INFERENCE_TEMP_BEV_FOLDER)
    return bev_image

def main():
    pass

if __name__ == "__main__":
    main()
