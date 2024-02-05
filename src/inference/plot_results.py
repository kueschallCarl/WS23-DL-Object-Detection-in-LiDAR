import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import tkinter as tk
from PIL import Image, ImageTk

class ImageDisplayApp:
    """
    A class representing a tkinter-based GUI application for displaying images and drawing bounding boxes.

    Attributes:
        root (tk.Tk): The main tkinter window.
        current_image (tk.PhotoImage): The currently displayed image.
        canvas (tk.Canvas): The canvas for displaying images and bounding boxes.
        change_image_button (tk.Button): A button to change the displayed image.
    """

    def __init__(self, root):
        """
        Initialize the ImageDisplayApp.

        Args:
            root (tk.Tk): The main tkinter window.
        """
        self.root = root
        self.root.title("Image Display App")
        
        # Initialize an empty image
        self.current_image = None
        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        # Add a button to trigger the function that changes the image
        self.change_image_button = tk.Button(root, text="Change Image", command=self.change_image)
        self.change_image_button.pack()

    def change_image(self):
        """
        Change the displayed image on the canvas.

        Replace this with your own image loading logic.
        """
        # For example, you can use the Image.open() from the PIL library
        new_image_path = "path_to_your_image.jpg"
        new_image = Image.open(new_image_path)

        # Convert the image to a format compatible with tkinter
        self.current_image = ImageTk.PhotoImage(new_image)

        # Update the canvas size to match the image dimensions
        self.canvas.config(width=self.current_image.width(), height=self.current_image.height())

        # Clear previous drawings on the canvas
        self.canvas.delete("all")

        # Display the new image on the canvas
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_image)

    def draw_rectangles(self, bboxes_data):
        """
        Draw bounding boxes on the canvas.

        Args:
            bboxes_data (dict): A dictionary containing bounding box information.
        """
        # Draw bounding boxes on the canvas
        for idx, bbox_info in bboxes_data.items():
            x = bbox_info["x"] * self.current_image.width()
            y = bbox_info["y"] * self.current_image.height()
            w = bbox_info["w"] * self.current_image.width()
            h = bbox_info["h"] * self.current_image.height()

            self.canvas.create_rectangle(x - w/2, y - h/2, x + w/2, y + h/2, outline="red", width=2)

            # Add confidence score as text
            self.canvas.create_text(x - w/2, y - h/2 - 15, text=f"Conf: {bbox_info['confidence']:.2f}", fill="red")

def plot_bounding_box_single_image(bboxes_data, image_folder, filename, app):
    """
    Plot bounding boxes on a single image in a tkinter window.

    Args:
        bboxes_data (dict): A dictionary containing bounding box information.
        image_folder (str): The folder containing image files.
        filename (str): The filename of the image to be displayed.
        app (ImageDisplayApp): An instance of the ImageDisplayApp class.
    """
    # Load the image
    image_path = f"{image_folder}/{filename.split('.')[0]}.png"
    img = Image.open(image_path)

    # Update the displayed image in the tkinter window
    new_image_tk = ImageTk.PhotoImage(img)
    app.current_image = new_image_tk

    # Update the canvas size to match the image dimensions
    app.canvas.config(width=app.current_image.width(), height=app.current_image.height())

    # Clear previous drawings on the canvas
    app.canvas.delete("all")

    # Display the new image on the canvas
    app.canvas.create_image(0, 0, anchor="nw", image=app.current_image)

    # Draw rectangles on the canvas
    app.draw_rectangles(bboxes_data)

    # Update the tkinter window
    app.root.update()

def plot_bounding_boxes(json_file_path, image_folder):
    """
    Plot bounding boxes on multiple images using matplotlib.

    Args:
        json_file_path (str): The path to a JSON file containing bounding box data.
        image_folder (str): The folder containing image files.
    """
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

def main():
    """
    The main function that runs the script.
    """
    json_file_path = r"inference_results\First_Test_Inference_2024-02-03_19-14-48\results_First_Test_Inference_2024-02-03_19-14-48.json"
    image_folder = r"temp/bev_images"
    plot_bounding_boxes(json_file_path, image_folder)

if __name__ == "__main__":
    main()
