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


def extract_coordinates(geometry):
    if hasattr(geometry, '__len__') and len(geometry) == 2:
        top_left, bottom_right = geometry
        x1, y1 = top_left
        x2, y2 = bottom_right
        coordinates = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]
        return coordinates
    return []

def load_model(model_path, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    predictor = ocr_predictor(
        det_arch=config['det_arch'],
        reco_arch=config['reco_arch'],
        pretrained=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor.load_state_dict(torch.load(model_path, map_location=device))
    predictor.eval()

    if torch.cuda.is_available():
        predictor = predictor.cuda()

    return predictor

def detect_text(predictor, image_path, text_threshold=0.7):
    doc = DocumentFile.from_images(image_path)

    with torch.no_grad():
        result = predictor(doc)

    boxes = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                coordinates = extract_coordinates(line.geometry)

                if len(coordinates) == 4:
                    box = np.array(coordinates)

                    if hasattr(line, 'confidence') and line.confidence >= text_threshold:
                        boxes.append(box)
                    elif not hasattr(line, 'confidence'):
                        boxes.append(box)

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    pixel_boxes = []
    for box in boxes:
        pixel_box = box.copy()
        pixel_box[:, 0] *= width
        pixel_box[:, 1] *= height
        pixel_boxes.append(pixel_box)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, np.array(pixel_boxes), np.array(pixel_boxes)

def show_boxes(image, boxes):
    img_show = image.copy()
    for box in boxes:
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_show, [box], isClosed=True, color=(0, 255, 0), thickness=2)
    plt.figure(figsize=(12, 12))
    plt.imshow(img_show)
    plt.axis('off')
    plt.show()

def crop_id_card(image, boxes):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crops = []

    for poly in boxes:
        poly = np.array(poly).astype(np.int32)

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        # Apply mask
        masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

        # Get bounding box and crop
        x, y, w, h = cv2.boundingRect(poly)
        cropped = masked[y:y+h, x:x+w]
        crops.append(cropped)

    return crops


def plot_crops(crops, per_row=5, size=4):

    n = len(crops)
    rows = (n + per_row - 1) // per_row  # Ceiling division

    fig, axs = plt.subplots(rows, per_row, figsize=(per_row * size, rows * size))

    # Flatten axs for easy indexing, handle 1D or 2D array of axes
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i in range(len(axs)):
        if i < n:
            axs[i].imshow(crops[i])
            axs[i].set_title(f"Crop #{i+1}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

# Define the classifier saved model architecture
class LanguageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LanguageClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class_names = ['Arabic', 'Frensh']

def predict_language_cnn(image, model, class_names, transform):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence = confidence.item() * 100

    return predicted_class, confidence

def classify_crops(crops):
    results = []

    for i, crop in enumerate(crops):
        try:
            prediction, confidence = predict_language_cnn(crop, model, class_names, test_transform)
            results.append((prediction, confidence))
            print(f"Crop {i+1}: {prediction} ({confidence:.2f}%)")

        except Exception as e:
            print(f"Error processing crop {i+1}: {str(e)}")
            results.append(("Error", 0))

    return results

def plot_classification(original_image, crops, boxes, classification_results):
    img_with_boxes = original_image.copy()

    for i, (box, (prediction, confidence)) in enumerate(zip(boxes, classification_results)):
        box = box.astype(np.int32).reshape((-1, 1, 2))

        # Colors: Green for Arabic, Blue for English, Red for errors
        color = (0, 255, 0) if prediction == 'Arabic' else (255, 0, 0) if prediction == 'Frensh' else (0, 0, 255)

        # Draw thicker box
        cv2.polylines(img_with_boxes, [box], isClosed=True, color=color, thickness=3)

        # Get top-left corner for text placement
        x_min = np.min(box[:, 0, 0])
        y_min = np.min(box[:, 0, 1])

        # Create background for text
        label = f"{prediction} {confidence:.0f}%"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Draw text background
        cv2.rectangle(img_with_boxes,
                     (x_min, y_min - text_size[1] - 5),
                     (x_min + text_size[0] + 10, y_min),
                     color, -1)

        # Draw text
        cv2.putText(img_with_boxes, label, (x_min + 5, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Language Classification Results')
    plt.show()

def run_ocr(image, predictor):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        result = predictor([image_rgb])

    ocr_results = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])

                if line.words:
                    x_coords = []
                    y_coords = []
                    for word in line.words:
                        for point in word.geometry:
                            x_coords.append(point[0])
                            y_coords.append(point[1])

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    ocr_results.append({
                        'text': line_text,
                        'bbox': (x_min, y_min, x_max, y_max),
                        'confidence': sum(word.confidence for word in line.words) / len(line.words)
                    })

    return ocr_results

def plot_ocr_results(image, ocr_results):
    img_with_text = image.copy()
    height, width = image.shape[:2]

    for i, result in enumerate(ocr_results):
        text = result['text']
        confidence = result['confidence']
        x_min, y_min, x_max, y_max = result['bbox']

        # Convert normalized coordinates to pixels
        x_min_px = int(x_min * width)
        y_min_px = int(y_min * height)
        x_max_px = int(x_max * width)
        y_max_px = int(y_max * height)

        # Draw bounding box
        cv2.rectangle(img_with_text, (x_min_px, y_min_px), (x_max_px, y_max_px), (0, 255, 255), 2)

        # Draw text background
        label = f"{text} ({confidence:.2f})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        cv2.rectangle(img_with_text,
                     (x_min_px, y_min_px - text_size[1] - 5),
                     (x_min_px + text_size[0] + 10, y_min_px),
                     (0, 255, 255), -1)

        # Draw text
        cv2.putText(img_with_text, label, (x_min_px + 5, y_min_px - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('OCR Results - All Detected Text')
    plt.show()

def display_ocr_results(ocr_results):
    print("=== OCR RESULTS ===")
    print(f"Found {len(ocr_results)} text lines:")
    print("-" * 50)

    for i, result in enumerate(ocr_results, 1):
        print(f"{i}. '{result['text']}'")
        print(f"   Confidence: {result['confidence']:.3f}")
        print()

def filter_ocr_with_classification(ocr_results, image, model, class_names, transform, confidence_threshold=0.7):
    filtered_texts = []
    height, width = image.shape[:2]

    for i, ocr_item in enumerate(ocr_results):
        text = ocr_item['text']
        ocr_confidence = ocr_item['confidence']
        x_min, y_min, x_max, y_max = ocr_item['bbox']

        # Skip empty text
        if not text.strip():
            continue

        # Convert normalized coordinates to pixels
        x_min_px = int(x_min * width)
        y_min_px = int(y_min * height)
        x_max_px = int(x_max * width)
        y_max_px = int(y_max * height)

        # Extract the region from image
        region = image[y_min_px:y_max_px, x_min_px:x_max_px]

        if region.size == 0:
            continue

        # Classify the region
        predicted_class, classification_confidence = predict_language_cnn(region, model, class_names, transform)

        # Keep text based on classification rules
        if ocr_confidence < confidence_threshold:
            # Low OCR confidence - use classification to decide
            if predicted_class == 'Arabic' and classification_confidence > 70:
                # Low confidence OCR + high confidence Arabic classification = discard
                print(f"Discarding: '{text}' (OCR: {ocr_confidence:.3f}, Class: {predicted_class} {classification_confidence:.1f}%)")
                continue
            else:
                # Keep if French or uncertain
                filtered_texts.append({
                    'text': text,
                    'ocr_confidence': ocr_confidence,
                    'classification': predicted_class,
                    'classification_confidence': classification_confidence,
                    'bbox': ocr_item['bbox'],
                    'region_id': i + 1
                })
        else:
            # High OCR confidence - keep regardless of classification
            filtered_texts.append({
                'text': text,
                'ocr_confidence': ocr_confidence,
                'classification': predicted_class,
                'classification_confidence': classification_confidence,
                'bbox': ocr_item['bbox'],
                'region_id': i + 1
            })

    return filtered_texts

def display_filtered_ocr_results(filtered_texts):
    print("=== FILTERED OCR RESULTS ===")
    print(f"Total texts: {len(filtered_texts)}")
    print("-" * 50)

    french_texts = [item for item in filtered_texts if item['classification'] == 'French']
    arabic_texts = [item for item in filtered_texts if item['classification'] == 'Arabic']
    other_texts = [item for item in filtered_texts if item['classification'] not in ['French', 'Arabic']]

    print(f"French texts: {len(french_texts)}")
    print(f"Arabic texts: {len(arabic_texts)}")
    print(f"Other/Uncertain: {len(other_texts)}")
    print("-" * 50)

    if french_texts:
        print("\n FRENCH TEXT (KEEP):")
        for result in french_texts:
            print(f"Region {result['region_id']}: '{result['text']}'")
            print(f"  OCR confidence: {result['ocr_confidence']:.3f}")
            print(f"  Classification: {result['classification']} ({result['classification_confidence']:.1f}%)")
            print()

    if arabic_texts:
        print("\n ARABIC TEXT (FILTERED):")
        for result in arabic_texts:
            print(f"Region {result['region_id']}: '{result['text']}'")
            print(f"  OCR confidence: {result['ocr_confidence']:.3f}")
            print(f"  Classification: {result['classification']} ({result['classification_confidence']:.1f}%)")
            print()

    if other_texts:
        print("\n OTHER TEXT (KEEP):")
        for result in other_texts:
            print(f"Region {result['region_id']}: '{result['text']}'")
            print(f"  OCR confidence: {result['ocr_confidence']:.3f}")
            print(f"  Classification: {result['classification']} ({result['classification_confidence']:.1f}%)")
            print()

def plot_final_results(image, filtered_texts):
    img_with_text = image.copy()
    height, width = image.shape[:2]

    for result in filtered_texts:
        text = result['text']
        x_min, y_min, x_max, y_max = result['bbox']
        language = result['classification']

        x_min_px = int(x_min * width)
        y_min_px = int(y_min * height)
        x_max_px = int(x_max * width)
        y_max_px = int(y_max * height)

        # Color based on language
        color = (0, 0, 255) if language == 'French' else (0, 255, 0)  # Red for French, Green for others

        cv2.rectangle(img_with_text, (x_min_px, y_min_px), (x_max_px, y_max_px), color, 2)

        label = f"{language}: {text}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        cv2.rectangle(img_with_text,
                     (x_min_px, y_min_px - text_size[1] - 5),
                     (x_min_px + text_size[0] + 10, y_min_px),
                     color, -1)

        cv2.putText(img_with_text, label, (x_min_px + 5, y_min_px - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Final Filtered OCR Results')
    plt.show()