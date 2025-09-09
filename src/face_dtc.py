import sys
import os
import cv2
import numpy as np
import unicodedata
import string

from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageFilter

import re
import torch
from datetime import datetime
import json
import difflib

from doctr.models import ocr_predictor
from doctr.io import DocumentFile

import torch.nn as nn
from torchvision import transforms


from ultralytics import YOLO

def box_contains(boxA, boxB):
    return (boxA[0] <= boxB[0]) and (boxA[1] <= boxB[1]) and (boxA[2] >= boxB[2]) and (boxA[3] >= boxB[3])

def detect_face(image_path, model_path='/content/yolov8x.pt', conf=0.25,
                                        edge_threshold=20, max_width_ratio=0.55):
    model = YOLO(model_path)
    results = model(image_path, conf=conf)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    img_width = image.shape[1]
    img_center_x = img_width / 2

    to_remove = set()
    n = len(boxes)

    for i in range(n):
        x0, _, x1, _ = boxes[i]
        box_width = x1 - x0

        if x0 <= edge_threshold and x1 >= (img_width - edge_threshold):
            to_remove.add(i)
            continue

        if box_width > max_width_ratio * img_width:
            to_remove.add(i)
            continue

        for j in range(n):
            if i != j:
                if box_contains(boxes[i], boxes[j]):
                    to_remove.add(i)

    filtered_boxes = [boxes[i] for i in range(n) if i not in to_remove]

    if not filtered_boxes:
        return None, None

    best_face = None
    best_crop = None
    min_dist = float('inf')

    for box in filtered_boxes:
        x0, y0, x1, y1 = box
        face_center_x = (x0 + x1) / 2

        if face_center_x < img_center_x:
            dist = img_center_x - face_center_x
            if dist < min_dist:
                min_dist = dist
                best_face = [x0, y0, x1, y1]
                best_crop = image[y0:y1, x0:x1]

    return best_face, best_crop