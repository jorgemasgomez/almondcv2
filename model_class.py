from email.mime import image
import glob
import os
import pickle
import shutil
import sys
import tkinter as tk
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import segmentation_refinement as refine
import torch
import yaml
from PIL import Image, ImageTk
from sahi.predict import get_sliced_prediction, AutoDetectionModel
from sahi.slicing import slice_image
from torchvision.ops import nms
from ultralytics import YOLO, SAM, settings
from ultralytics.engine.results import Results, Boxes

settings.update({"wandb": False})

try:
    from PySide6.QtCore import QRectF, Qt, Signal
    from PySide6.QtGui import QColor, QCursor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QDialogButtonBox,
        QGraphicsEllipseItem,
        QGraphicsPixmapItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

# =============================================================================
# Annotation conversion utilities
# =============================================================================

def polygons_to_bboxes(input_folder, output_folder):
    """
    Convert YOLO polygon annotation files into YOLO bounding box annotation files.

    Each input TXT file is expected to contain lines in the following format:

        class_id x1 y1 x2 y2 x3 y3 ...

    where the polygon coordinates are normalized YOLO coordinates.

    The output files will contain bounding boxes in YOLO format:

        class_id x_center y_center width height

    Parameters
    ----------
    input_folder : str
        Folder containing the polygon TXT annotation files.

    output_folder : str
        Folder where the converted bounding box TXT files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))

    for txt_file in txt_files:
        output_lines = []

        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 3:
                    continue

                class_id = parts[0]
                coords = [float(x) for x in parts[1:]]

                xs = coords[::2]
                ys = coords[1::2]

                # Compute the minimum bounding rectangle around the polygon.
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)

                # Convert rectangle coordinates to YOLO bounding box format.
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                output_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}"
                )

        # Save the converted annotation file using the original file name.
        base_name = os.path.basename(txt_file)
        output_file = os.path.join(output_folder, base_name)

        with open(output_file, "w") as f:
            f.write("\n".join(output_lines))

    print(f"Conversion completed. Bounding boxes saved in: {output_folder}")


# =============================================================================
# Mask refinement utilities
# =============================================================================

def process_single_mask(mask, box, image, H, W, refiner, pad=5):
    """
    Refine a single segmentation mask using a cropped region around its bounding box.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask corresponding to one detected object.

    box : list or tuple
        Bounding box coordinates in the format [x1, y1, x2, y2].

    image : np.ndarray
        Original image.

    H : int
        Image height.

    W : int
        Image width.

    refiner : object
        Segmentation refinement model/object with a `.refine()` method.

    pad : int, optional
        Padding added around the bounding box before cropping, by default 5.

    Returns
    -------
    dict
        Dictionary containing the original bounding box and the refined contour.
    """
    # Convert bounding box coordinates to integers.
    x1, y1, x2, y2 = map(int, box)

    # Apply padding while keeping the crop inside the image boundaries.
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(W, x2 + pad)
    ry2 = min(H, y2 + pad)

    # Crop image and mask around the object.
    image_crop = image[ry1:ry2, rx1:rx2]
    mask_crop = (mask[ry1:ry2, rx1:rx2] * 255).astype(np.uint8)

    # Refine the cropped mask.
    # L=200 provides a good balance between speed and quality for almond images.
    refined_crop = refiner.refine(
        image_crop,
        mask_crop,
        fast=False,
        L=200
    )

    # Reconstruct the refined mask in the original image size.
    mask_refined = np.zeros((H, W), dtype=np.uint8)
    mask_refined[ry1:ry2, rx1:rx2] = refined_crop

    # Extract contours from the refined mask.
    contours, _ = cv2.findContours(
        mask_refined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return {"box": box, "contour": None}

    # Keep only the largest contour.
    main_contour = max(contours, key=cv2.contourArea)

    return {
        "box": box,
        "contour": main_contour
    }


# =============================================================================
# Visualization utilities
# =============================================================================

def draw_image_with_masks(image, image_masks, discard_ids=None, image_name=None):
    """
    Draw segmentation masks, contours, and object IDs over an image.

    Parameters
    ----------
    image : np.ndarray
        Original image.

    image_masks : list of dict
        List of dictionaries containing at least a `"contour"` key.

    discard_ids : list, set or None, optional
        Object indices to display as discarded. These IDs are zero-based.

    image_name : str or None, optional
        Optional image name. Currently not used, but kept for compatibility.

    Returns
    -------
    np.ndarray
        Annotated image with masks, contours, and labels.
    """
    image_annotated = image.copy()

    # Convert discard IDs to a set for faster lookup.
    discard_ids = set(discard_ids) if discard_ids is not None else set()

    for idx, item in enumerate(image_masks, start=1):
        contour = item.get("contour", None)

        if contour is None or len(contour) == 0:
            continue

        # Internal object IDs are zero-based, while displayed labels start at 1.
        is_discarded = (idx - 1) in discard_ids

        # Create a binary mask from the contour.
        mask = np.zeros(image_annotated.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        # Create a color overlay depending on the object status.
        overlay = np.zeros_like(image_annotated)

        if is_discarded:
            # Discarded objects are displayed in blue.
            overlay[:, :, 0] = mask
            alpha = 0.4
            contour_color = (255, 0, 0)      # Blue in BGR
            text_color = (200, 200, 200)     # Light gray
        else:
            # Valid objects are displayed with a red mask and green contour.
            overlay[:, :, 2] = mask
            alpha = 0.5
            contour_color = (0, 255, 0)      # Green in BGR
            text_color = (255, 255, 0)       # Cyan/yellowish in BGR

        # Blend overlay with the original image.
        image_annotated = cv2.addWeighted(
            image_annotated,
            1.0,
            overlay,
            alpha,
            0
        )

        # Draw the object contour.
        cv2.drawContours(
            image_annotated,
            [contour],
            -1,
            contour_color,
            2
        )

        # Draw the object ID label near the first contour point.
        x, y = contour[0][0]

        cv2.putText(
            image_annotated,
            str(idx),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            text_color,
            2
        )

    return image_annotated


def get_contour_center(contour):
    """
    Compute the centroid of a contour using image moments.

    Parameters
    ----------
    contour : np.ndarray
        Object contour.

    Returns
    -------
    tuple
        Centroid coordinates as `(cx, cy)`.
        Returns `(0, 0)` if the contour area is zero.
    """
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return 0, 0

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return cx, cy


# =============================================================================
# Mouse interaction callbacks
# =============================================================================

def mouse_callback(event, x, y, flags, param):
    """
    Store clicked points when the left mouse button is pressed.

    Parameters
    ----------
    event : int
        OpenCV mouse event.

    x, y : int
        Mouse coordinates.

    flags : int
        OpenCV mouse event flags.

    param : dict
        Dictionary containing a `"points"` list.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param["points"].append((x, y))


def draw_rectangle(event, x, y, flags, param):
    """
    Mouse callback to draw a rectangle interactively.

    The rectangle starts when the left mouse button is pressed,
    updates while the mouse moves, and finishes when the button is released.

    Parameters
    ----------
    event : int
        OpenCV mouse event.

    x, y : int
        Mouse coordinates.

    flags : int
        OpenCV mouse event flags.

    param : dict
        Dictionary containing rectangle drawing state.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param["drawing"] = True
        param["start"] = (x, y)
        param["end"] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and param["drawing"]:
        param["end"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        param["drawing"] = False
        param["end"] = (x, y)
        param["bbox_done"] = True


def roi_clicks(event, x, y, flags, param):
    """
    Mouse callback to collect positive and negative points inside a displayed ROI.

    Left click adds a positive point.
    Right click adds a negative point.

    The clicked display coordinates are mapped back to the original image
    coordinates using the stored bounding box and display size.

    Parameters
    ----------
    event : int
        OpenCV mouse event.

    x, y : int
        Mouse coordinates inside the displayed ROI.

    flags : int
        OpenCV mouse event flags.

    param : dict
        Dictionary containing ROI information, points, and labels.
    """
    if param["bbox_coords"] is None:
        return

    bx1, by1, bx2, by2 = param["bbox_coords"]

    # Convert coordinates from displayed ROI space to original image space.
    px = bx1 + int(x * (bx2 - bx1) / param["display_w"])
    py = by1 + int(y * (by2 - by1) / param["display_h"])

    if event == cv2.EVENT_LBUTTONDOWN:
        param["points"].append((px, py))
        param["labels"].append(1)  # Positive point

    elif event == cv2.EVENT_RBUTTONDOWN:
        param["points"].append((px, py))
        param["labels"].append(0)  # Negative point


# =============================================================================
# General helper functions
# =============================================================================

def chunk_list(lst, n):
    """
    Split a list into consecutive chunks of size `n`.

    Parameters
    ----------
    lst : list
        Input list.

    n : int
        Chunk size.

    Yields
    ------
    list
        Consecutive chunks of the original list.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def navigate_image(parent=None):
    """
    Display a small Tkinter window to navigate between images.

    Parameters
    ----------
    parent : tk.Tk or tk.Toplevel, optional
        Parent Tkinter window. If None, a hidden root window is created.

    Returns
    -------
    str
        Selected action: `"next"`, `"prev"`, or `"quit"`.
    """
    result = {"action": None}

    def on_next():
        result["action"] = "next"
        win.destroy()

    def on_prev():
        result["action"] = "prev"
        win.destroy()

    def on_quit():
        result["action"] = "quit"
        win.destroy()

    # Create a Toplevel window so the main root window remains untouched.
    if parent is None:
        root = tk.Tk()
        root.withdraw()
        parent = root

    win = tk.Toplevel(parent)
    win.title("Navigate images")
    win.geometry("300x100")
    win.attributes("-topmost", True)

    tk.Label(win, text="Choose an action:").pack(pady=5)

    tk.Button(
        win,
        text="Next →",
        command=on_next
    ).pack(side="left", padx=5, pady=5)

    tk.Button(
        win,
        text="← Previous",
        command=on_prev
    ).pack(side="left", padx=5, pady=5)

    tk.Button(
        win,
        text="Exit",
        command=on_quit
    ).pack(side="left", padx=5, pady=5)

    win.mainloop()

    return result["action"]


# =============================================================================
# Object ordering utilities
# =============================================================================

def group_by_rows(image_masks, y_threshold=30):
    """
    Group segmented objects into rows based on their contour centroids.

    Objects with similar Y-coordinate centroids are assigned to the same row.

    Parameters
    ----------
    image_masks : list of dict
        List of mask dictionaries containing a `"contour"` key.

    y_threshold : int, optional
        Maximum vertical distance between centroids to consider objects
        part of the same row, by default 30.

    Returns
    -------
    list
        List of rows, where each row contains tuples of:
        `(item, cx, cy)`.
    """
    if len(image_masks) == 0:
        return []

    items = []

    for item in image_masks:
        contour = item.get("contour", None)

        if contour is None:
            continue

        cx, cy = get_contour_center(contour)
        items.append((item, cx, cy))

    if len(items) == 0:
        return []

    # Sort objects from top to bottom.
    items.sort(key=lambda x: x[2])

    rows = []
    current_row = [items[0]]

    for item in items[1:]:
        _, _, cy = item
        _, _, previous_cy = current_row[-1]

        if abs(cy - previous_cy) < y_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]

    rows.append(current_row)

    return rows


def sort_rows(rows):
    """
    Sort objects row by row from left to right.

    Parameters
    ----------
    rows : list
        List of rows generated by `group_by_rows()`.

    Returns
    -------
    list
        Flattened list of items sorted from top to bottom and left to right.
    """
    sorted_items = []

    for row in rows:
        # Sort each row from left to right according to the X centroid.
        row_sorted = sorted(row, key=lambda x: x[1])

        # Keep only the original item dictionary.
        sorted_items.extend([item[0] for item in row_sorted])

    return sorted_items


# =============================================================================
# Bounding box utilities
# =============================================================================

def expand_boxes(boxes, img_shape, margin=0.03):
    """
    Expand bounding boxes by a relative margin.

    Parameters
    ----------
    boxes : list or np.ndarray
        Bounding boxes in the format `[x1, y1, x2, y2]`.

    img_shape : tuple
        Shape of the image. Usually `image.shape`.

    margin : float, optional
        Relative expansion margin based on box width and height, by default 0.03.

    Returns
    -------
    np.ndarray
        Expanded bounding boxes clipped to the image boundaries.
    """
    h, w = img_shape[:2]
    new_boxes = []

    for x1, y1, x2, y2 in boxes:
        box_width = x2 - x1
        box_height = y2 - y1

        dx = int(box_width * margin)
        dy = int(box_height * margin)

        nx1 = max(0, x1 - dx)
        ny1 = max(0, y1 - dy)
        nx2 = min(w, x2 + dx)
        ny2 = min(h, y2 + dy)

        new_boxes.append([nx1, ny1, nx2, ny2])

    return np.array(new_boxes)


# =============================================================================
# Image splitting utility
# =============================================================================

def split_image(image, max_dim):
    """
    Split an image into smaller parts if its dimensions exceed `max_dim`.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    max_dim : int
        Maximum allowed width or height for each split image.

    Returns
    -------
    list
        List of image crops or tiles.

    Notes
    -----
    This function is currently a placeholder because the original code
    was incomplete.
    """
    raise NotImplementedError("The body of split_image() was not provided.")
    h, w = image.shape[:2]

    tiles = []
    coords = []

    step = max_dim  # puedes cambiar a 0.8*max_dim si quieres overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = image[y:y+step, x:x+step]
            tiles.append(tile)
            coords.append((x, y))

    return tiles, coords



class ModelSegmentation():
    def __init__(self, working_directory):
        """
        Initializes the model with the specified working directory and sets the device to either CPU or GPU.
        
        Parameters:
            working_directory (str): The directory where the input data and results will be stored.
        
        Returns:
            None
        """
        self.working_directory = working_directory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if torch.cuda.is_available():
            gpu_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_index)
            gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory
            gpu_memory_gb = gpu_memory / (1024 ** 3)
            print(f"Detected GPU: {gpu_name}")
            print(f"Total GPU Memory: {gpu_memory_gb:.2f} GB")
        else:
            print("No GPU detected. Using CPU.")

    def train_segmentation_model(self, input_zip, pre_model="yolov8n-seg.pt", epochs=100, imgsz=640, batch=-1, name_segmentation="",
                                 retina_masks=True, colab=False):
        """
        Trains the segmentation model using a YOLO segmentation file.

        Parameters:
            input_zip (str): Path to the input zip file containing images and annotations.
            pre_model (str): Path to the pretrained model (default: "yolov8n-seg.pt").
            epochs (int): Number of epochs for training (default: 100).
            imgsz (int): Image size for training (default: 640).
            batch (int): Batch size for training (default: -1, auto batch size).
            name_segmentation (str): Name for the segmentation model's output (default: "").
            retina_masks (bool): Whether to use retina masks for segmentation (default: True).
            
            

        Returns:
            results_test_set (list): List of results containing the predicted masks for the test set.
        """
        input_zip_no_extension, extension = os.path.splitext(input_zip)
        output_folder_zip = os.path.join(self.working_directory, input_zip_no_extension)
        self.output_folder_zip = output_folder_zip

        if os.path.exists(self.output_folder_zip):
            shutil.rmtree(self.output_folder_zip)
        os.makedirs(self.output_folder_zip, exist_ok=True)

        zip_file_path = os.path.join(self.working_directory, input_zip)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_folder_zip)

        yaml_file = os.path.join(self.output_folder_zip, "data.yaml")
        self.yaml_file = yaml_file

        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        modified_data = {
            'path': self.output_folder_zip,
            'train': 'images/Train',
            'val': 'images/Validation',
            'test': 'images/Test'
        }
        if 'names' in data:
            modified_data['names'] = data['names']

        with open(self.yaml_file, 'w') as file:
            yaml.dump(modified_data, file, default_flow_style=False)

        results_models_directory = os.path.join(self.working_directory, f"results_models_segmentation_{name_segmentation}")
        self.results_models_directory = results_models_directory
        os.makedirs(self.results_models_directory, exist_ok=True)

        if colab:
            model = YOLO(pre_model)
            model.to(self.device)
            model.train(data=self.yaml_file, epochs=epochs, imgsz=imgsz, batch=batch, project=name_segmentation, name="results_training")
            shutil.move(name_segmentation, results_models_directory)
            
            test_set_folder = os.path.join(self.output_folder_zip, "images/Test/")
            self.test_set_folder = test_set_folder

            results_test_set = model.predict(self.test_set_folder, imgsz=imgsz, show=False, save=True, show_boxes=False, project=self.results_models_directory, save_txt=True,
                                            name="predictions_test", retina_masks=retina_masks)
        else:
            model = YOLO(pre_model)
            model.to(self.device)
            model.train(data=self.yaml_file, epochs=epochs, imgsz=imgsz, batch=batch, project=self.results_models_directory, name="results_training")

            test_set_folder = os.path.join(self.output_folder_zip, "images/Test/")
            self.test_set_folder = test_set_folder

            results_test_set = model.predict(self.test_set_folder, imgsz=imgsz, show=False, save=True, show_boxes=False, project=self.results_models_directory, save_txt=True,
                                            name="predictions_test", retina_masks=retina_masks)
        return results_test_set

    def predict_model(self, model_path, folder_input, imgsz=640, check_result=False, conf=0.6, max_det=300, retina_mask=True):
        """
        Predicts segmentation masks using a trained model.

        Parameters:
            model_path (str): Path to the trained model file.
            folder_input (str): Path to the folder containing images to be segmented.
            imgsz (int): Image size for prediction (default: 640).
            check_result (bool): Whether to save the results for further inspection (default: False).
            conf (float): Confidence threshold for predictions (default: 0.6).
            max_det (int): Maximum number of detections per image (default: 300).
            retina_mask (bool): Whether to use retina masks for segmentation (default: True).

        Returns:
            results (list): List of predictions for each image in the input folder. Each entry contains masks and coordinates of segmented objects.
        """
        model = YOLO(model_path)
        if not check_result:
            results = model.predict(folder_input, imgsz=imgsz, show=False, save=False, show_boxes=False, conf=conf, max_det=max_det, retina_masks=retina_mask)
        else:
            results = model.predict(folder_input, imgsz=imgsz, show=False, save=True, show_boxes=False, project=self.working_directory, name="check_results", conf=conf, max_det=max_det, retina_masks=retina_mask)

        return results

    def predict_model_sahi(self, model_path, folder_input=None, confidence_treshold=0.5, model_type='yolov8',
                            slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, postprocess_type="NMS", check_result=False
                            , postprocess_match_metric="IOS", postprocess_match_threshold=0.5, retina_masks=True, imgsz=640, image_array=None):
        """
        Predicts segmentation masks using the SAHI method (Slice and Heal Inference) for large images.

        Parameters:
            model_path (str): Path to the trained model file.
            folder_input (str): Path to the folder containing images to be segmented.
            image_array : Option for a direct picture array
            confidence_treshold (float): Confidence threshold for predictions (default: 0.5).
            model_type (str): Type of YOLO model to use (default: "yolov8").
            slice_height (int): Height of each slice (default: 640).
            slice_width (int): Width of each slice (default: 640).
            overlap_height_ratio (float): Overlap ratio in height between slices (default: 0.2).
            overlap_width_ratio (float): Overlap ratio in width between slices (default: 0.2).
            postprocess_type (str): Type of postprocessing for results (default: "NMS").
            check_result (bool): Whether to save the results for inspection (default: False).
            postprocess_match_metric (str): Metric to use for postprocessing (default: "IOS").
            postprocess_match_threshold (float): Threshold for postprocessing (default: 0.5).
            retina_masks (bool): Whether to use retina masks (default: True).
            imgsz (int): Image size for prediction (default: 640).

        Returns:
            results_list (list): List of results for each image processed, including segmented masks and additional information.
        """
        detection_model_seg = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=confidence_treshold,
            device=self.device,
            retina_masks=retina_masks,
            image_size=imgsz)
        
        if folder_input is not None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_list = [os.path.join(folder_input, file)
                        for file in os.listdir(folder_input)
                        if os.path.splitext(file)[1].lower() in image_extensions]
        elif image_array is not None:
            image_list = [image_array]
        else:
            raise ValueError("Provide a folder or a picture")

        results_list = []
        i = 1
        for pic in image_list:
            print(f"Pic {i}/{len(image_list)}")

            try:
                result = get_sliced_prediction(
                    image=pic, detection_model=detection_model_seg, slice_height=slice_height,
                    slice_width=slice_width, overlap_height_ratio=overlap_height_ratio, overlap_width_ratio=overlap_width_ratio,
                    postprocess_type=postprocess_type, postprocess_match_metric=postprocess_match_metric,
                    postprocess_match_threshold=postprocess_match_threshold, perform_standard_pred=True)
            except Exception as e:
                print(f"Error processing segmentation image {pic}: {e}")
                continue

            torch.cuda.empty_cache()
            results_list.append([result, pic])
            i += 1
            if check_result:
                pic_sin_ext = os.path.splitext(os.path.basename(pic))[0]
                check_result_path = os.path.join(self.working_directory, "check_results")
                os.makedirs(check_result_path, exist_ok=True)
                result.export_visuals(export_dir=check_result_path, hide_labels=True, rect_th=1, file_name=f"prediction_result_{pic_sin_ext}")
        return results_list
    
    def slice_predict_reconstruct(self, imgsz, model_path, slice_width, slice_height, overlap_height_ratio, 
                                overlap_width_ratio, input_folder=None, conf=0.5, retina_mask=True, image_array=None):
        """
        Slices images, runs segmentation on all slices in batch (GPU), and reconstructs the full mask.
        """
        if input_folder is not None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_list = [os.path.join(input_folder, file)
                        for file in os.listdir(input_folder)
                        if os.path.splitext(file)[1].lower() in image_extensions]
        elif image_array is not None:
            image_list = [image_array]
        else:
            raise ValueError("Provide a folder or a picture")

        mask_list_images = []
        n = 1

        # Cargar modelo una sola vez
        model = YOLO(model_path, verbose=False).to(self.device)

        for image_path in image_list:
            if image_array is None:
                print(f"Processing Image {n}/{len(image_list)}")
                image_selected = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            else:
                image_selected = image_path

            if image_selected.shape[2] == 4:
                image_selected = cv2.cvtColor(image_selected, cv2.COLOR_RGBA2RGB)

            # Slice the image into chunks
            image_sliced = slice_image(image=image_selected, slice_width=slice_width,
                                    slice_height=slice_height, overlap_height_ratio=overlap_height_ratio,
                                    overlap_width_ratio=overlap_width_ratio, verbose=True)
            mask_complete = np.zeros((image_sliced.original_image_height, image_sliced.original_image_width), dtype=np.uint8)
            # Si image_sliced.images es una lista de arrays de NumPy
            slices_batch = image_sliced.images  # Ya estÃ¡n en formato NumPy

            # Aseguramos que se pase sin conversiÃ³n a tensores
            with torch.no_grad():  # Evita cÃ¡lculos innecesarios de gradientes
                results = model.predict(slices_batch, imgsz=imgsz, conf=conf, retina_masks=retina_mask, verbose=False)

            for i, result in enumerate(results):
                h_slice, w_slice = image_sliced.images[i].shape[:2]
                start_x, start_y = image_sliced.starting_pixels[i]

                mask_combined_slice = np.zeros((h_slice, w_slice), dtype=np.uint8)

                if result.masks and result.masks.data is not None:
                    for mask in result.masks.data:
                        mask = mask.cpu().numpy() * 255
                        mask = cv2.resize(mask, (w_slice, h_slice))
                        mask_combined_slice = cv2.bitwise_or(mask_combined_slice, mask.astype(np.uint8))

                mask_added = np.zeros_like(mask_complete)
                mask_added[start_y:start_y + h_slice, start_x:start_x + w_slice] = mask_combined_slice
                mask_complete = cv2.bitwise_or(mask_complete, mask_added)

            mask_list_images.append([mask_complete, image_path])
            n += 1

        return mask_list_images
    
    def train_detection_model(self, input_zip, pre_model="yolov8n.pt", epochs=100, imgsz=640, batch=-1, name_detection=""):
        """
        Trains a YOLO detection model. Automatically converts polygon masks to bounding boxes.

        Parameters:
            input_zip (str): Path to the input zip file containing images and annotations.
            pre_model (str): Path to the pretrained YOLO model (default: "yolov8n.pt").
            epochs (int): Number of epochs for training (default: 100).
            imgsz (int): Image size for training (default: 640).
            batch (int): Batch size for training (default: -1, auto batch size).
            name_detection (str): Name for the detection model's output folder (default: "").

        Returns:
            results_test_set (list): Predictions for the test set.
        """
        input_zip_no_ext, _ = os.path.splitext(input_zip)
        output_folder_zip = os.path.join(self.working_directory, input_zip_no_ext)
        self.output_folder_zip = output_folder_zip

        if os.path.exists(self.output_folder_zip):
            shutil.rmtree(self.output_folder_zip)
        os.makedirs(self.output_folder_zip, exist_ok=True)

        # Extract zip
        zip_file_path = os.path.join(self.working_directory, input_zip)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_folder_zip)

        # Convert polygon masks to bounding boxes
        labels_folders = ["images/Train", "images/Validation", "images/Test"]
        for folder in labels_folders:
            label_folder = os.path.join(self.output_folder_zip, folder.replace("images", "labels"))
            if os.path.exists(label_folder):
                temp_folder = label_folder + "_bbox"
                polygons_to_bboxes(label_folder, temp_folder)
                # Replace original folder with converted bounding boxes
                shutil.rmtree(label_folder)
                os.rename(temp_folder, label_folder)

        # Prepare YAML
        yaml_file = os.path.join(self.output_folder_zip, "data.yaml")
        self.yaml_file = yaml_file
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        modified_data = {
            'path': self.output_folder_zip,
            'train': 'images/Train',
            'val': 'images/Validation',
            'test': 'images/Test'
        }
        if 'names' in data:
            modified_data['names'] = data['names']

        with open(yaml_file, 'w') as file:
            yaml.dump(modified_data, file, default_flow_style=False)

        # Create results folder
        results_models_directory = os.path.join(self.working_directory, f"results_models_detection_{name_detection}")
        self.results_models_directory = results_models_directory
        os.makedirs(results_models_directory, exist_ok=True)

        # Train YOLO detection model
        model = YOLO(pre_model)
        model.to(self.device)
        model.train(data=yaml_file, epochs=epochs, imgsz=imgsz, batch=batch, project=results_models_directory, name="results_training")

        # Predict on test set
        test_set_folder = os.path.join(self.output_folder_zip, "images/Test/")
        self.test_set_folder = test_set_folder

        results_test_set = model.predict(
            test_set_folder,
            imgsz=imgsz,
            show=False,
            save=True,
            show_boxes=True,
            project=results_models_directory,
            save_txt=True,
            name="predictions_test",
            retina_masks=False  # bounding boxes, no masks
        )

        return results_test_set
      
    def predict_detection_sam(
        self,
        model_path,
        folder_input,
        sam_path="sam_b.pt",
        imgsz=640,
        conf=0.6,
        max_det=30000,
        save_results=True,
        output_name="sam_results",
        retina_masks= True,
        batch_size=1,
        margin_bbox=0.03,
        max_dimension=2800,# for splitting the picture for the prediction
        nms_largeimages=0.4
           
    ):
        """
        Runs YOLO detection first, then refines each bounding box using Ultralytics SAM.

        Parameters:
            model_path (str): Path to YOLO detection model.
            folder_input (str): Folder with input images.
            sam_path (str): SAM model name or path (e.g., "sam_b.pt").
            imgsz (int): Image size for YOLO.
            conf (float): Confidence threshold.
            max_det (int): Max detections per image.
            save_results (bool): Whether to save masks and overlays.
            output_name (str): Output folder name.
            batch_size (int): Batch size for YOLO prediction.

        Returns:
            all_results (list): List with masks and boxes per image.
        """
        # -------------------------
        # Load models
        # -------------------------
        model = YOLO(model_path)
        model.to(self.device)

        sam_model = SAM(sam_path)
        sam_model.to(self.device)
        refiner = refine.Refiner(device='cuda:0') 



        # -------------------------
        # Prepare output folders
        # -------------------------
        output_folder = os.path.join(self.working_directory, output_name)



        # -------------------------
        # Get image paths
        # -------------------------
        valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
        image_paths = [
            os.path.join(folder_input, f)
            for f in os.listdir(folder_input)
            if f.lower().endswith(valid_ext)
        ]

        # -------------------------
        # Run YOLO detection
        # -------------------------
        
        normal_images = []
        large_images = []

        for path in image_paths:
            img = cv2.imread(path)
            h, w = img.shape[:2]

            if max(h, w) > max_dimension:
                large_images.append(path)
            else:
                normal_images.append(path)
        yolo_results = []

        for chunk in chunk_list(normal_images, 50):
            results= model.predict(
                chunk,
                imgsz=imgsz,
                conf=conf,
                batch=batch_size,
                max_det=max_det,
                save=False,
                show=False,
                verbose=True,
                stream=True
            )
            for r, p in zip(results, chunk):
                yolo_results.append((p, r))


        for path in large_images:
            try:
                image = cv2.imread(path)
                h, w = image.shape[:2]

                tiles, coords = split_image(image, max_dimension)

                all_boxes = []
                all_scores = []
                all_cls = []

                # -------------------------
                # predict per tile
                # -------------------------
                for tile, (x_off, y_off) in zip(tiles, coords):

                    res = model.predict(
                        tile,
                        imgsz=imgsz,
                        conf=conf,
                        batch=1,
                        max_det=max_det,
                        save=False,
                        show=False,
                        verbose=False
                    )[0]

                    if res.boxes is None or len(res.boxes) == 0:
                        continue

                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy()

                    # offset a global
                    boxes[:, [0, 2]] += x_off
                    boxes[:, [1, 3]] += y_off

                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_cls.append(cls)

                # -------------------------
                # merge
                # -------------------------
                boxes_np = np.vstack(all_boxes)
                scores_np = np.hstack(all_scores)
                cls_np = np.hstack(all_cls)

                if boxes_np.shape[0] == 0:
                    continue

                boxes = torch.tensor(np.vstack(all_boxes))
                scores = torch.tensor(np.hstack(all_scores))
                cls = torch.tensor(np.hstack(all_cls))

                keep = nms(boxes, scores, nms_largeimages)

                boxes = boxes[keep]
                scores = scores[keep]
                cls = cls[keep]

                # -------------------------
                # rebuild YOLO result
                # -------------------------
                data = torch.cat([
                    boxes,
                    scores.unsqueeze(1),
                    cls.unsqueeze(1)
                ], dim=1)

                dummy = Results(
                    orig_img=image,
                    path=path,
                    names=model.model.names
                )

                dummy.boxes = Boxes(data, image.shape[:2])

                yolo_results.append((path, dummy))
            except:
                print(f"Error processing large image {path}")
                continue
                



        all_results = []
        # -------------------------
        # Process each image
        # -------------------------
        for image_path, result in yolo_results:
            
            print(image_path)
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"âš ï¸ Could not read image: {image_path}")
                    continue

                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                image_masks = []

                base_name = os.path.splitext(os.path.basename(image_path))[0]

                # -------------------------
                # Process each bounding box with SAM
                # -------------------------
                boxes = expand_boxes(boxes, image.shape, margin=margin_bbox)  # prueba 0.1â€“0.3
                boxes_input = boxes.tolist()

                sam_results = sam_model(image, bboxes=boxes_input, retina_masks=retina_masks)


                image_masks = []


                masks=sam_results[0].masks.data.cpu().numpy()
                H, W = image.shape[:2]

                # Usamos max_workers bajo (ej. 4) para no saturar la GPU

                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Quitamos la 'i' de aquÃ­ para que solo pasemos 6 argumentos
                    futures = [
                        executor.submit(process_single_mask, masks[idx], boxes_input[idx], image, H, W, refiner)
                        for idx in range(len(masks))
                    ]
                    
                    # Recogemos los resultados
                    image_masks = [f.result() for f in futures]
                
                rows = group_by_rows(image_masks, y_threshold=40)  # ajusta este valor
                image_masks= sort_rows(rows)


                all_results.append([image_path,image_masks])


            except Exception as e:
                    print(f"âŒ Error processing {image_path}: {e}")
                    continue

        print(f"âœ… Finished. Results saved in: {output_folder}")

        self.all_results=all_results
        if save_results:
            with open(f"{self.working_directory}/all_results_backup.pkl", "wb") as f:
                pickle.dump(self.all_results, f)

        return all_results
    
    def check_segmentation_twosteps(self, sam_path):
        print("Starting check_segmentation_twosteps...")
        
        # Make sure Qt is initialized.
        _ensure_qt_app()
    
        if not hasattr(self, 'all_results') or len(self.all_results) == 0:
            print("Error: self.all_results is not defined or is empty.")
            return
        
        print("Loading SAM model...")
        sam_model = SAM(sam_path)
        sam_model.to(self.device)
        print("SAM model loaded.")
        
        print("Loading refiner...")
        refiner = refine.Refiner(device='cuda:0')
        print("Refiner loaded.")
    
        pic_id = 0
        

        txt_path = os.path.join(self.working_directory, "discard_morphology_session.txt")

        self.discard_morphology = defaultdict(list)

        if os.path.exists(txt_path):
            print(f"Loading discard_morphology from: {txt_path}")

            with open(txt_path, "r") as f:
                next(f)  # Skip header.

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    img_name, idx = line.split("\t")

                    # Subtract 1 to convert to internal 0-based indexing.
                    self.discard_morphology[img_name].append(int(idx) - 1)

        else:
            print("TXT file does not exist. Initializing an empty discard_morphology dictionary.")
            self.discard_morphology = defaultdict(list)
    
        while 0 <= pic_id < len(self.all_results):
            print(f"Processing image {pic_id + 1}/{len(self.all_results)}: {self.all_results[pic_id][0]}")
            pic = self.all_results[pic_id]
            image_path = pic[0]
            image_masks = pic[1]
    
            # Main loop: display the image with action buttons until the user finishes.
            action_loop = True
            while action_loop:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image: {image_path}")
                    break
    
                rows = group_by_rows(image_masks, y_threshold=40)
                image_masks = sort_rows(rows)
                discard_ids = self.discard_morphology.get(os.path.basename(image_path), [])
                image_annotated = draw_image_with_masks(image, image_masks, discard_ids)
    
                # Display image with action buttons.
                viewer = ZoomCanvasDialog(
                    None,
                    image_annotated,
                    title=f"Image {pic_id + 1}/{len(self.all_results)} - Select action",
                    mode="points",
                    allow_negative=False,
                    show_action_buttons=True
                )
                clicked_points = viewer.show() or []
                selected_action = viewer.selected_action
    
                print(f"Selected action: {selected_action}, Points: {clicked_points}")
    
                # ===== REMOVE DUPLICATES =====
                if selected_action == "delete_duplicates":
                    if not clicked_points:
                        QMessageBox.warning(None, "Warning", "You must click on the image to select overlapping points.")
                        continue
                    
                    print("Processing: Remove duplicates...")
                    indices_to_delete = set()
                    indices_to_keep = set()
    
                    for (px, py) in clicked_points:
                        overlapping = []
                        for idx, item in enumerate(image_masks):
                            contour = item.get("contour", None)
                            if contour is None or len(contour) == 0:
                                continue
                            inside = cv2.pointPolygonTest(contour, (int(px), int(py)), False)
                            if inside >= 0:
                                area = cv2.contourArea(contour)
                                overlapping.append((idx, area))
    
                        if len(overlapping) > 1:
                            indices_human = [i + 1 for i, _ in overlapping]
                            print(f"Overlap at ({px},{py}): objects {indices_human}")
                            overlapping_sorted = sorted(overlapping, key=lambda x: x[1], reverse=True)
                            keep_idx = overlapping_sorted[0][0]
                            indices_to_keep.add(keep_idx)
                            for idx, _ in overlapping_sorted[1:]:
                                indices_to_delete.add(idx)
    
                    indices_to_delete = indices_to_delete - indices_to_keep
                    if indices_to_delete:
                        indices_sorted_display = sorted([i + 1 for i in indices_to_delete])
                        print(f"Removing duplicates: {indices_sorted_display}")
                        image_masks = [item for i, item in enumerate(image_masks) if i not in indices_to_delete]
                    else:
                        print("No duplicates were found.")
    
                # ===== DELETE OBJECTS =====
                elif selected_action == "delete":
                    if not clicked_points:
                        QMessageBox.warning(None, "Warning", "You must click on the objects you want to delete.")
                        continue
                    
                    print("Processing: Delete objects...")
                    indices_to_delete = set()
                    for idx, item in enumerate(image_masks):
                        contour = item.get("contour", None)
                        if contour is None or len(contour) == 0:
                            continue
                        for (px, py) in clicked_points:
                            inside = cv2.pointPolygonTest(contour, (int(px), int(py)), False)
                            if inside >= 0:
                                indices_to_delete.add(idx)
                                break
    
                    if indices_to_delete:
                        indices_sorted = sorted([i + 1 for i in indices_to_delete])
                        print(f"Deleting objects: {indices_sorted}")
                        image_masks = [item for i, item in enumerate(image_masks) if i not in indices_to_delete]
                    else:
                        print("No object was selected.")
    
                # ===== ADD OBJECT =====
                elif selected_action == "add":
                    print("Processing: Add object...")
                    # Display canvas in bbox mode to draw the object box.
                    viewer_bbox = ZoomCanvasDialog(
                        None,
                        image_annotated,
                        title="Draw the bbox of the object to add",
                        mode="bbox",
                        allow_negative=False
                    )
                    bbox = viewer_bbox.show()
    
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        image_crop = image[y1:y2, x1:x2]
                        
                        if image_crop is not None and image_crop.size > 0:
                            # Display the ROI and collect SAM points.
                            roi_view = ZoomCanvasDialog(
                                None,
                                image_crop,
                                title="ROI - Select SAM points",
                                mode="points",
                                allow_negative=True
                            )
                            roi_points = roi_view.show() or []
    
                            points_crop = []
                            labels = []
                            for item in roi_points:
                                if len(item) == 2:
                                    px, py = item
                                    label = 1
                                else:
                                    px, py, label = item
                                points_crop.append((px, py))
                                labels.append(int(label))
    
                            if len(points_crop) > 0:
                                sam_results = sam_model(image_crop, points=points_crop, labels=labels)
                                mask_crop = sam_results[0].masks.data.cpu().numpy()[0]
                                mask_crop = (mask_crop * 255).astype(np.uint8)
    
                                # Ask whether the mask should be refined.
                                ask_refine = QMessageBox.question(
                                    None,
                                    "Refinement",
                                    "Do you want to refine the mask?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No
                                ) == QMessageBox.Yes
    
                                if ask_refine:
                                    refined_crop = refiner.refine(image_crop, mask_crop, fast=False, L=900)
                                else:
                                    refined_crop = mask_crop
    
                                H, W = image.shape[:2]
                                mask_refined = np.zeros((H, W), dtype=np.uint8)
                                mask_refined[y1:y2, x1:x2] = refined_crop
    
                                contours, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    main_contour = max(contours, key=cv2.contourArea)
                                    image_masks.append({
                                        "box": (x1, y1, x2, y2),
                                        "contour": main_contour
                                    })
                                    print("Object added successfully.")
                                else:
                                    print("Could not find a contour.")
                            else:
                                print("No points were selected.")
                        else:
                            print("Empty ROI.")
                    else:
                        print("No bbox was drawn.")

                elif selected_action == "mark_morphology_discard":
                    if not clicked_points:
                        QMessageBox.warning(None, "Warning", "Click on the almonds to discard.")
                        continue

                    print("Marking almonds to be discarded for morphology...")

                    for (px, py) in clicked_points:
                        for idx, item in enumerate(image_masks):
                            contour = item.get("contour", None)
                            if contour is None:
                                continue

                            inside = cv2.pointPolygonTest(contour, (int(px), int(py)), False)
                            if inside >= 0:
                                self.discard_morphology[os.path.basename(image_path)].append(idx)
                                print(f"Marked {image_path} -> ID {idx}")
                                break
    
                # ===== DONE / NEXT IMAGE =====
                elif selected_action is None:
                    # The user pressed Enter/Done without selecting an action.
                    action_loop = False
                    print("Saving changes and moving to the next image...")
    
                    output_txt = os.path.join(self.working_directory, "discard_morphology_session.txt")

                    # ---------------------------
                    # 1. Load existing rows.
                    # ---------------------------
                    existing = set()

                    if os.path.exists(output_txt):
                        with open(output_txt, "r") as f:
                            next(f)  # Header.
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue

                                img_name, idx = line.split("\t")
                                existing.add((img_name, int(idx) - 1))  # Convert to 0-based indexing.

                    # ---------------------------
                    # 2. Add new rows.
                    # ---------------------------
                    new_rows = set()

                    for img_name, ids in self.discard_morphology.items():
                        for idx in ids:
                            new_rows.add((img_name, idx))

                    # ---------------------------
                    # 3. Merge without duplicates.
                    # ---------------------------
                    all_rows = existing | new_rows

                    # Sort rows by image name and object index.
                    all_rows = sorted(all_rows, key=lambda x: (x[0], x[1]))

                    # ---------------------------
                    # 4. Write clean TXT file.
                    # ---------------------------
                    with open(output_txt, "w") as f:
                        f.write("image\tdiscard_id\n")

                        for img_name, idx in all_rows:
                            f.write(f"{img_name}\t{idx + 1}\n")

                    print(f"Clean TXT updated: {output_txt}")
    
            goto_number = ""  # Number currently being typed.
            goto_mode = False  # True when the user is in "go to image" mode.

            self.all_results[pic_id] = (image_path, image_masks)
            with open(f"{self.working_directory}/all_results_backup.pkl", "wb") as f:
                pickle.dump(self.all_results, f)

            # Create a black display image.
            display = np.zeros((200, 1000, 3), dtype=np.uint8)

            # Main control text.
            text = f"D: next | A: previous | S: save | Q: quit | G: go to | {pic_id + 1}/{len(self.all_results)}"
            cv2.putText(display, text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Control", display)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord('d') or key == ord('D'):
                pic_id += 1
            if key == ord('a') or key == ord('A'):
                pic_id -= 1
            if key == ord('s') or key == ord('S'):
                # Save action.
                pass
            if key == ord('q') or key == ord('Q'):
                break
            if key == ord('g') or key == ord('G'):
                # Enable go-to mode.
                goto_mode = True
                goto_number = ""

                # Small loop to type the target image number.
                while goto_mode:
                    # Create a black display window.
                    display = np.zeros((200, 1000, 3), dtype=np.uint8)
                    cv2.putText(display, text, (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                    # Display the number currently being typed.
                    if goto_number != "":
                        cv2.putText(display, f"Go to: {goto_number}", (20, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow("Control", display)
                    k = cv2.waitKey(0) & 0xFF

                    if 48 <= k <= 57:  # Numbers.
                        goto_number += chr(k)
                    elif k == 8:  # Backspace.
                        goto_number = goto_number[:-1]
                    elif k == ord('r') or k == ord('R'):  # Ready.
                        if goto_number != "":
                            pic_id = int(goto_number) - 1  # Update pic_id.
                        goto_mode = False
                        goto_number = ""
                    elif k == ord('c'):  # Cancel.
                        goto_mode = False
                        goto_number = ""

            # Keep pic_id within the valid range.
            pic_id = max(0, min(pic_id, len(self.all_results) - 1))
    
        return self.all_results


class ZoomCanvasDialog:
    """
    Tkinter viewer with:
    - preserved aspect ratio
    - mouse-wheel zoom
    - middle-button pan
    - clicks using real image coordinates
    - bbox mode with drag selection
    """

    def __init__(self, root, image_bgr, title="Viewer", mode="points", allow_negative=False):
        self.root = root
        self.mode = mode
        self.allow_negative = allow_negative

        self.image_bgr = image_bgr.copy()
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = self.image_rgb.shape[:2]

        self.points = []          # [(x, y)] or [(x, y, label)]
        self.bbox = None          # (x1, y1, x2, y2)
        self.finished = False
        self.cancelled = False

        # Transformation
        self.zoom = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 20.0
        self.pan_x = 0
        self.pan_y = 0

        # Pan
        self._pan_start = None

        # BBox
        self._bbox_start = None
        self._bbox_current = None
        self._drawing_bbox = False

        self.win = tk.Toplevel()
        self.win.title(title)

        # Do not use fullscreen.
        self.win.geometry("1200x800")
        self.win.minsize(900, 700)

        self.win.configure(bg="black")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.canvas = tk.Canvas(self.win, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.tk_img = None  # Important.

        self.win.update_idletasks()
        self.win.after(50, self._redraw)  # Key step: delayed rendering.

        # Bottom help bar.
        self.help_var = tk.StringVar()
        help_frame = tk.Frame(self.win, bg="gray15")
        help_frame.place(relx=0, rely=1, anchor="sw", relwidth=1.0)
        help_label = tk.Label(
            help_frame,
            textvariable=self.help_var,
            fg="white",
            bg="gray15",
            font=("Arial", 11),
            anchor="w",
            padx=10,
            pady=4
        )
        help_label.pack(fill="x")

        self._set_help_text()

        # Events.
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)   # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel)     # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)     # Linux scroll down

        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._do_pan)

        if self.mode == "bbox":
            self.canvas.bind("<ButtonPress-1>", self._bbox_press)
            self.canvas.bind("<B1-Motion>", self._bbox_motion)
            self.canvas.bind("<ButtonRelease-1>", self._bbox_release)
        else:
            self.canvas.bind("<Button-1>", self._left_click)
            if self.allow_negative:
                self.canvas.bind("<Button-3>", self._right_click)

        self.canvas.focus_set()

        # At the end of __init__:
        self.win.update_idletasks()  # Force Tkinter to calculate widget sizes.
        self._redraw()

    def _set_help_text(self):
        if self.mode == "bbox":
            self.help_var.set(
                "BBox mode: drag with the left mouse button to draw the box. "
                "Mouse wheel = zoom | middle button = pan | Enter/Q = accept | Esc = cancel"
            )
        else:
            if self.allow_negative:
                self.help_var.set(
                    "Points mode: left click = positive, right click = negative. "
                    "Mouse wheel = zoom | middle button = pan | Enter/Q = accept | Esc = cancel"
                )
            else:
                self.help_var.set(
                    "Points mode: left click to add points. "
                    "Mouse wheel = zoom | middle button = pan | Enter/Q = accept | Esc = cancel"
                )

    def _current_scale(self):
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        base_scale = min(canvas_w / self.img_w, canvas_h / self.img_h)
        return base_scale * self.zoom

    def _image_origin(self):
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        scale = self._current_scale()

        disp_w = int(self.img_w * scale)
        disp_h = int(self.img_h * scale)

        x0 = (canvas_w - disp_w) // 2 + self.pan_x
        y0 = (canvas_h - disp_h) // 2 + self.pan_y

        return x0, y0, disp_w, disp_h, scale

    def _canvas_to_image(self, cx, cy):
        x0, y0, _, _, scale = self._image_origin()
        ix = int((cx - x0) / scale)
        iy = int((cy - y0) / scale)

        return ix, iy

    def _image_to_canvas(self, ix, iy):
        x0, y0, _, _, scale = self._image_origin()
        cx = int(x0 + ix * scale)
        cy = int(y0 + iy * scale)

        return cx, cy

    def _on_resize(self, event=None):
        if self.canvas.winfo_width() > 10 and self.canvas.winfo_height() > 10:
            self._redraw()

    def _on_mousewheel(self, event):
        # Zoom around the cursor position.
        if hasattr(event, "delta") and event.delta != 0:
            factor = 1.1 if event.delta > 0 else 0.9
        else:
            factor = 1.1 if event.num == 4 else 0.9

        old_scale = self._current_scale()
        mouse_x = event.x
        mouse_y = event.y
        ix, iy = self._canvas_to_image(mouse_x, mouse_y)

        new_zoom = self.zoom * factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        self.zoom = new_zoom

        new_scale = self._current_scale()

        # Keep the point under the cursor fixed.
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        new_disp_w = int(self.img_w * new_scale)
        new_disp_h = int(self.img_h * new_scale)

        desired_x0 = mouse_x - ix * new_scale
        desired_y0 = mouse_y - iy * new_scale

        self.pan_x = desired_x0 - ((canvas_w - new_disp_w) // 2)
        self.pan_y = desired_y0 - ((canvas_h - new_disp_h) // 2)

        self._redraw()

    def _start_pan(self, event):
        self._pan_start = (event.x, event.y)

    def _do_pan(self, event):
        if self._pan_start is None:
            return

        x_prev, y_prev = self._pan_start
        dx = event.x - x_prev
        dy = event.y - y_prev

        self.pan_x += dx
        self.pan_y += dy
        self._pan_start = (event.x, event.y)

        self._redraw()

    def _left_click(self, event):
        ix, iy = self._canvas_to_image(event.x, event.y)

        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            self.points.append((ix, iy))
            self._redraw()

    def _right_click(self, event):
        ix, iy = self._canvas_to_image(event.x, event.y)

        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            self.points.append((ix, iy, 0))
            self._redraw()

    def _bbox_press(self, event):
        ix, iy = self._canvas_to_image(event.x, event.y)

        if 0 <= ix < self.img_w and 0 <= iy < self.img_h:
            self._bbox_start = (ix, iy)
            self._bbox_current = (ix, iy)
            self._drawing_bbox = True
            self._redraw()

    def _bbox_motion(self, event):
        if not self._drawing_bbox or self._bbox_start is None:
            return

        ix, iy = self._canvas_to_image(event.x, event.y)
        ix = max(0, min(self.img_w - 1, ix))
        iy = max(0, min(self.img_h - 1, iy))

        self._bbox_current = (ix, iy)
        self._redraw()

    def _bbox_release(self, event):
        if not self._drawing_bbox or self._bbox_start is None:
            return

        ix, iy = self._canvas_to_image(event.x, event.y)
        ix = max(0, min(self.img_w - 1, ix))
        iy = max(0, min(self.img_h - 1, iy))

        x1, y1 = self._bbox_start
        x2, y2 = ix, iy

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:
            self.bbox = (x1, y1, x2, y2)

        self._drawing_bbox = False
        self._bbox_start = None
        self._bbox_current = None

        self._redraw()

    def _redraw(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if w < 50 or h < 50:
            self.win.after(50, self._redraw)
            return

        self.canvas.delete("all")

        # Base scale plus zoom.
        canvas_w = w
        canvas_h = h

        base_scale = min(canvas_w / self.img_w, canvas_h / self.img_h)
        scale = base_scale * self.zoom

        disp_w = max(1, int(self.img_w * scale))
        disp_h = max(1, int(self.img_h * scale))

        # Resize image.
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(self.image_rgb, (disp_w, disp_h), interpolation=interp)

        self.tk_img = ImageTk.PhotoImage(Image.fromarray(resized))

        # Always center the image before applying pan.
        x0 = (canvas_w - disp_w) // 2 + self.pan_x
        y0 = (canvas_h - disp_h) // 2 + self.pan_y

        self.canvas.create_image(x0, y0, anchor="nw", image=self.tk_img)

        # Points.
        for item in self.points:
            if len(item) == 2:
                ix, iy = item
                label = 1
            else:
                ix, iy, label = item

            cx = x0 + ix * scale
            cy = y0 + iy * scale

            color = "yellow" if label == 1 else "red"
            self.canvas.create_oval(
                cx - 4,
                cy - 4,
                cx + 4,
                cy + 4,
                fill=color,
                outline=color
            )

        # Debug text.
        self.canvas.create_text(
            10,
            10,
            anchor="nw",
            text=f"Zoom: {self.zoom:.2f}",
            fill="cyan"
        )

        self.canvas.focus_set()

    def accept(self):
        self.finished = True
        self.close()

    def close(self):
        if not self.cancelled and not self.finished:
            self.finished = True

        try:
            self.win.destroy()
        except:
            pass

    def show(self):
        self.win.update_idletasks()

        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()

        width = min(1400, max(900, int(screen_w * 0.8)))
        height = min(900, max(700, int(screen_h * 0.8)))

        pos_x = max(0, (screen_w - width) // 2)
        pos_y = max(0, (screen_h - height) // 2)

        self.win.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

        self.win.deiconify()
        self.win.lift()
        self.win.attributes("-topmost", True)
        self.win.focus_force()
        self.win.grab_set()  # Block the parent window.

        self._redraw()
        self.win.update_idletasks()
        self.win.update()

        self.win.wait_window()  # Wait for the window to close.

        return self.bbox if self.mode == "bbox" else self.points


def _ensure_qt_app():
    if not QT_AVAILABLE:
        raise ImportError("PySide6 is not available in this environment.")

    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

    return app


def _bgr_to_qpixmap(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    height, width, channels = image_rgb.shape
    bytes_per_line = channels * width

    qimage = QImage(
        image_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888
    ).copy()

    return QPixmap.fromImage(qimage)


class _QtImageView(QGraphicsView):
    point_added = Signal(object)
    bbox_changed = Signal(object)

    def __init__(self, image_bgr, mode="points", allow_negative=False, parent=None):
        super().__init__(parent)

        self.mode = mode
        self.allow_negative = allow_negative

        self.image_bgr = image_bgr.copy()
        self.img_h, self.img_w = self.image_bgr.shape[:2]

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QColor("black"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)

        self._pixmap_item = QGraphicsPixmapItem(_bgr_to_qpixmap(self.image_bgr))
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(QRectF(0, 0, self.img_w, self.img_h))

        self._point_items = []
        self._bbox_item = None
        self._preview_item = None
        self._bbox_start = None

        self._is_panning = False
        self._pan_origin = None
        self._user_zoomed = False
        self._zoom_factor = 1.0
        self._min_zoom = 0.2
        self._max_zoom = 25.0

    def fit_image(self):
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

        self._zoom_factor = 1.0
        self._user_zoomed = False

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if not self._user_zoomed:
            self.fit_image()

    def wheelEvent(self, event):
        angle = event.angleDelta().y()

        if angle == 0:
            return

        factor = 1.15 if angle > 0 else 1 / 1.15
        new_zoom = self._zoom_factor * factor

        if not (self._min_zoom <= new_zoom <= self._max_zoom):
            return

        self._zoom_factor = new_zoom
        self._user_zoomed = True

        self.scale(factor, factor)

    def mousePressEvent(self, event):
        button = event.button()
        scene_pos = self.mapToScene(event.position().toPoint())
        image_pos = self._scene_to_image(scene_pos)

        if button == Qt.MiddleButton:
            self._is_panning = True
            self._pan_origin = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if self.mode == "points":
            if button == Qt.LeftButton and image_pos is not None:
                x, y = image_pos
                self.point_added.emit((x, y, 1))
                event.accept()
                return

            if button == Qt.RightButton and self.allow_negative and image_pos is not None:
                x, y = image_pos
                self.point_added.emit((x, y, 0))
                event.accept()
                return

        if self.mode == "bbox" and button == Qt.LeftButton and image_pos is not None:
            self._bbox_start = image_pos
            self._set_preview_bbox((*image_pos, *image_pos))
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning and self._pan_origin is not None:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )

            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )

            event.accept()
            return

        if self.mode == "bbox" and self._bbox_start is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            image_pos = self._scene_to_image(scene_pos, clamp=True)

            if image_pos is not None:
                self._set_preview_bbox((*self._bbox_start, *image_pos))
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        button = event.button()

        if button == Qt.MiddleButton and self._is_panning:
            self._is_panning = False
            self._pan_origin = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if self.mode == "bbox" and button == Qt.LeftButton and self._bbox_start is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            image_pos = self._scene_to_image(scene_pos, clamp=True)

            if image_pos is not None:
                x1, y1 = self._bbox_start
                x2, y2 = image_pos

                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:
                    self.bbox_changed.emit((x1, y1, x2, y2))

            self._bbox_start = None
            self._clear_preview_bbox()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def set_points(self, points):
        for item in self._point_items:
            self._scene.removeItem(item)

        self._point_items = []

        for point in points:
            if len(point) == 2:
                x, y = point
                label = 1
            else:
                x, y, label = point

            radius = 8
            color = QColor("yellow") if label == 1 else QColor("red")

            item = QGraphicsEllipseItem(
                x - radius,
                y - radius,
                radius * 2,
                radius * 2
            )

            item.setPen(QPen(color, 2))
            item.setBrush(QColor(color.red(), color.green(), color.blue(), 90))

            self._scene.addItem(item)
            self._point_items.append(item)

    def set_bbox(self, bbox):
        if self._bbox_item is not None:
            self._scene.removeItem(self._bbox_item)
            self._bbox_item = None

        if bbox is None:
            return

        x1, y1, x2, y2 = bbox
        rect = QRectF(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        self._bbox_item = QGraphicsRectItem(rect)
        self._bbox_item.setPen(QPen(QColor("cyan"), 2))

        self._scene.addItem(self._bbox_item)

    def _scene_to_image(self, scene_pos, clamp=False):
        x = int(round(scene_pos.x()))
        y = int(round(scene_pos.y()))

        if clamp:
            x = max(0, min(self.img_w - 1, x))
            y = max(0, min(self.img_h - 1, y))

            return x, y

        if 0 <= x < self.img_w and 0 <= y < self.img_h:
            return x, y

        return None

    def _set_preview_bbox(self, bbox):
        x1, y1, x2, y2 = bbox

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        rect = QRectF(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        if self._preview_item is None:
            self._preview_item = QGraphicsRectItem(rect)

            preview_pen = QPen(QColor("white"), 2)
            preview_pen.setStyle(Qt.DashLine)

            self._preview_item.setPen(preview_pen)
            self._scene.addItem(self._preview_item)
        else:
            self._preview_item.setRect(rect)

    def _clear_preview_bbox(self):
        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None


class QtZoomCanvasDialog:
    """
    Qt-based modal viewer for zoom, pan, points, and bbox selection.
    """

    def __init__(self, root, image_bgr, title="Viewer", mode="points", allow_negative=False, show_action_buttons=False):
        self.root = root
        self.mode = mode
        self.allow_negative = allow_negative
        self.show_action_buttons = show_action_buttons
        self.selected_action = None

        self.points = []
        self.bbox = None
        self.finished = False
        self.cancelled = False

        self._app = _ensure_qt_app()

        self._dialog = QDialog()
        self._dialog.setWindowTitle(title)
        self._dialog.setModal(True)
        self._dialog.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint
        )
        self._dialog.setMinimumSize(1000, 700)

        self._place_on_active_screen()

        layout = QVBoxLayout(self._dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.view = _QtImageView(
            image_bgr,
            mode=mode,
            allow_negative=allow_negative
        )
        layout.addWidget(self.view, 1)

        self.help_label = QLabel(self._help_text())
        self.help_label.setStyleSheet(
            "color: white; background: #2a2a2a; padding: 6px;"
        )
        layout.addWidget(self.help_label)

        # Action buttons, if enabled.
        if self.show_action_buttons:
            action_layout = QHBoxLayout()
            action_layout.setSpacing(8)

            btn_duplicates = QPushButton("Remove duplicates")
            btn_duplicates.setStyleSheet(
                "background-color: #FF6B6B; color: white; font-weight: bold; padding: 8px;"
            )
            btn_duplicates.clicked.connect(self._on_delete_duplicates)
            action_layout.addWidget(btn_duplicates)

            btn_delete = QPushButton("Delete object")
            btn_delete.setStyleSheet(
                "background-color: #FF4444; color: white; font-weight: bold; padding: 8px;"
            )
            btn_delete.clicked.connect(self._on_delete)
            action_layout.addWidget(btn_delete)

            btn_add = QPushButton("Add object")
            btn_add.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
            )
            btn_add.clicked.connect(self._on_add)
            action_layout.addWidget(btn_add)

            btn_morph = QPushButton("Discard morphology")
            btn_morph.setStyleSheet(
                "background-color: #FFB300; color: black; font-weight: bold; padding: 8px;"
            )
            btn_morph.clicked.connect(self._on_morphology_discard)
            action_layout.addWidget(btn_morph)

            action_widget = QWidget()
            action_widget.setLayout(action_layout)
            layout.addWidget(action_widget)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        self.button_box.button(QDialogButtonBox.Ok).setText("Done")
        self.button_box.button(QDialogButtonBox.Cancel).setText("Cancel")

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)

        layout.addWidget(self.button_box)

        self.view.point_added.connect(self._handle_point_added)
        self.view.bbox_changed.connect(self._handle_bbox_changed)

        self._shortcut_enter = QShortcut(QKeySequence("Return"), self._dialog)
        self._shortcut_enter.activated.connect(self.accept)

        self._shortcut_enter2 = QShortcut(QKeySequence("Enter"), self._dialog)
        self._shortcut_enter2.activated.connect(self.accept)

        self._shortcut_escape = QShortcut(QKeySequence("Escape"), self._dialog)
        self._shortcut_escape.activated.connect(self.close)

        self._shortcut_q = QShortcut(QKeySequence("Q"), self._dialog)
        self._shortcut_q.activated.connect(self.accept)

    def _help_text(self):
        if self.mode == "bbox":
            return (
                "BBox mode: drag with the left mouse button to draw the box. "
                "Mouse wheel = zoom | middle button = pan | Enter/Q = accept | Esc = cancel"
            )

        if self.allow_negative:
            return (
                "Points mode: left click = positive, right click = negative. "
                "Mouse wheel = zoom | middle button = pan | Enter/Q = accept | Esc = cancel"
            )

        return (
            "Points mode: left click to add points. "
            "Mouse wheel = zoom | middle button = pan | Enter/Q = accept | Esc = cancel"
        )

    def _on_morphology_discard(self):
        self.selected_action = "mark_morphology_discard"
        self.finished = True
        self._dialog.accept()

    def _on_delete_duplicates(self):
        """Callback for the 'Remove duplicates' button."""
        self.selected_action = "delete_duplicates"
        self.finished = True
        self._dialog.accept()

    def _on_delete(self):
        """Callback for the 'Delete object' button."""
        self.selected_action = "delete"
        self.finished = True
        self._dialog.accept()

    def _on_add(self):
        """Callback for the 'Add object' button."""
        self.selected_action = "add"
        self.finished = True
        self._dialog.accept()

    def _handle_point_added(self, point_data):
        x, y, label = point_data

        if label == 1:
            self.points.append((x, y))
        else:
            self.points.append((x, y, label))

        self.view.set_points(self.points)

    def _handle_bbox_changed(self, bbox):
        self.bbox = bbox
        self.view.set_bbox(self.bbox)

    def accept(self):
        self.finished = True
        self._dialog.accept()

    def close(self):
        self.cancelled = True
        self._dialog.reject()

    def show(self):
        self._place_on_active_screen()

        self.view.fit_image()

        self._dialog.showNormal()
        self._dialog.raise_()
        self._dialog.activateWindow()

        self._app.processEvents()
        self._dialog.exec()

        return self.bbox if self.mode == "bbox" else self.points

    def _place_on_active_screen(self):
        screen = QApplication.screenAt(QCursor.pos())

        if screen is None:
            screen = self._app.primaryScreen()

        if screen is None:
            return

        geometry = screen.availableGeometry()

        width = min(1500, max(1000, int(geometry.width() * 0.85)))
        height = min(950, max(700, int(geometry.height() * 0.85)))

        pos_x = geometry.x() + max(0, (geometry.width() - width) // 2)
        pos_y = geometry.y() + max(0, (geometry.height() - height) // 2)

        self._dialog.setGeometry(pos_x, pos_y, width, height)


if QT_AVAILABLE:
    ZoomCanvasDialog = QtZoomCanvasDialog