import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


def plot_bounding_boxes(json_file_path, image_folder):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    for file_name, bboxes_info in data.items():
        # Load the image
        image_path = os.path.join(image_folder, f"{file_name.split('.')[0]}.png")
        img = Image.open(image_path)

        # Create figure and axes
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Plot bounding boxes
        for idx, bbox_info in bboxes_info.items():
            x = bbox_info["x"] * img.width
            y = bbox_info["y"] * img.height
            w = bbox_info["w"] * img.width
            h = bbox_info["h"] * img.height

            rect = Rectangle((x - w/2, y - h/2), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add confidence score as text
            ax.text(x - w/2, y - h/2 - 5, f"Conf: {bbox_info['confidence']:.2f}", color='r')

        plt.title(f"Bounding Boxes for {file_name}")
        plt.show()


def plot_bounding_box_single_image(bboxes_data, image_folder, filename):
    # Load the image
    image_path = f"{image_folder}/{filename.split('.')[0]}.png"
    img = Image.open(image_path)

    # Create a drawing object for bounding boxes
    draw = ImageDraw.Draw(img)

    # Plot bounding boxes
    for idx, bbox_info in bboxes_data.items():
        x = int(bbox_info["x"] * img.width)
        y = int(bbox_info["y"] * img.height)
        w = int(bbox_info["w"] * img.width)
        h = int(bbox_info["h"] * img.height)

        draw.rectangle([(x - w//2, y - h//2), (x + w//2, y + h//2)], outline=(255, 0, 0), width=2)

        # Add confidence score as text
        draw.text((x - w//2, y - h//2 - 15), f"Conf: {bbox_info['confidence']:.2f}", fill=(255, 0, 0))

    # Save the image with bounding boxes
    output_folder = 'temp/'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'live_result.png')
    img.save(output_path)

    # Open the image in the default image viewer
    img.show()



def main():
    json_file_path = r"inference_results\First_Test_Inference_2024-02-03_19-14-48\results_First_Test_Inference_2024-02-03_19-14-48.json"
    image_folder = r"temp/bev_images"
    plot_bounding_boxes(json_file_path, image_folder)

if __name__ == "__main__":
    main()


"""
'inference_results\\First_Test_Inference_2024-02-03_19-14-48\\results_First_Test_Inference_2024-02-03_19-14-48.json'
'inference_results\First_Test_Inference_2024-02-03_19-21-08\results_First_Test_Inference_2024-02-03_19-21-08.json
"""