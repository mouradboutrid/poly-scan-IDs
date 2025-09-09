import sys
import os
import numpy as np
import unicodedata
import string
from collections import OrderedDict
import re
from datetime import datetime
import json
import difflib

def extract_prenom_pc(blocks, filtered_texts):
    def normalize(text):
        text = unicodedata.normalize("NFD", text.lower().strip())
        return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    def starts_with_label(text, label):
        return normalize(text).startswith(normalize(label))

    def is_in_filtered(text):
        return any(text.strip() == ft['text'].strip() for ft in filtered_texts)

    def find_label(labels, position_check=None):
        for i, block in enumerate(blocks):
            for label in labels:
                if starts_with_label(block['text'], label):
                    if not position_check or position_check(block['bbox']):
                        return block
        return None

    def get_block_above(ref_block):
        ref_y = ref_block['bbox'][1]
        candidates = [
            block for block in blocks
            if block['bbox'][3] < ref_y and block['bbox'][0] < 0.5
        ]
        return sorted(candidates, key=lambda b: -b['bbox'][3])[0] if candidates else None

    def get_blocks_below(ref_block, count):
        ref_y = ref_block['bbox'][3]
        candidates = [
            block for block in blocks
            if block['bbox'][1] > ref_y and block['bbox'][0] < 0.5
        ]
        return sorted(candidates, key=lambda b: b['bbox'][1])[:count]

    # Step 1: Try "Nom"
    for label_group in [['Nom/', 'Nom', 'Nom...'], ['Prénom/', 'Prénom', 'Prénom...']]:
        label_type = 'Nom' if 'Nom' in label_group[0] else 'Prénom'
        label_block = find_label(label_group)
        if label_block:
            above = get_block_above(label_block)
            if above and is_in_filtered(above['text']):
                return above['text']

            # Check below if Prénom case
            if label_type == 'Prénom':
                below = get_blocks_below(label_block, count=2)
                if below:
                    best = max(below, key=lambda b: b['ocr_confidence'])
                    if is_in_filtered(best['text']):
                        return best['text']

    # Step 2: Fallback to CONDUIRE header
    def is_top_left(bbox):
        x, y = bbox[0], bbox[1]
        return x < 0.5 and y < 0.3

    conduire_block = find_label(['PERMIS DE CONDUIRE'], position_check=is_top_left)
    if conduire_block:
        below = get_blocks_below(conduire_block, count=4)
        for b in below:
            if b['ocr_confidence'] > 0.87 and is_in_filtered(b['text']):
                return b['text']

    return None

def extract_nom_from_blocks_pc(blocks):
    keywords_sets = [
        {"Date", "Naissance", "Lieu"},
        {"Date", "Naissance"},
        {"Naissance"}
    ]

    x_mins = [block['bbox'][0] for block in blocks]
    x_maxs = [block['bbox'][2] for block in blocks]
    card_center_x = (min(x_mins) + max(x_maxs)) / 2

    blocks_sorted = sorted(blocks, key=lambda b: b['bbox'][1])

    def contains_keywords(text, keywords):
        text_lower = text.lower()
        return all(k.lower() in text_lower for k in keywords)

    def is_date(text):
        date_regex = re.compile(r'\b\d{2}[\./-]\d{2}[\./-]\d{4}\b')
        return bool(date_regex.search(text))

    for keywords in keywords_sets:
        for i, block in enumerate(blocks_sorted):
            if contains_keywords(block['text'], keywords):
                keyword_ymin = block['bbox'][1]
                candidate_blocks = [
                    b for b in blocks_sorted
                    if ((b['bbox'][0] + b['bbox'][2]) / 2) < card_center_x and b['bbox'][1] < keyword_ymin
                ]
                if candidate_blocks:
                    candidate_blocks = sorted(candidate_blocks, key=lambda b: b['bbox'][1], reverse=True)
                    return candidate_blocks[0]['text']
                else:
                    return blocks_sorted[i-1]['text'] if i > 0 else None

    first_date_block = None
    for block in blocks_sorted:
        if is_date(block['text']):
            first_date_block = block
            break

    if first_date_block:
        date_ymin = first_date_block['bbox'][1]
        above_blocks = [
            b for b in blocks_sorted
            if ((b['bbox'][0] + b['bbox'][2]) / 2) < card_center_x and b['bbox'][1] < date_ymin
        ]
        if above_blocks:
            above_blocks = sorted(above_blocks, key=lambda b: b['bbox'][1], reverse=True)
            block_above_date = above_blocks[0]
            block_above_date_ymin = block_above_date['bbox'][1]
            above_above_blocks = [
                b for b in blocks_sorted
                if ((b['bbox'][0] + b['bbox'][2]) / 2) < card_center_x and b['bbox'][1] < block_above_date_ymin
            ]
            if above_above_blocks:
                above_above_blocks = sorted(above_above_blocks, key=lambda b: b['bbox'][1], reverse=True)
                return above_above_blocks[0]['text']

    return None

def normalize_text_pc(text):
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def extract_lieu_de_naissance_pc(boxes):
    variants = [
        "délivré à", "delivre a", "delivré a", "délivre a", "dellivre a",
        "delivreà", "delivrea", "delivréà", "délivréà",
        "delivré a:", "délivré à:", "délivré à .", "délivré à :",
    ]

    date_pattern = re.compile(r'\b\d{2}/\d{2}/\d{4}\b')
    date_boxes = [b for b in boxes if date_pattern.search(b['text'])]

    if len(date_boxes) < 2:
        return None

    date_boxes = sorted(date_boxes, key=lambda b: b['bbox'][1])
    first_date = date_boxes[0]
    second_date = date_boxes[1]

    y_first_bottom = first_date['bbox'][3]
    y_second_top = second_date['bbox'][1]
    x_second_left = second_date['bbox'][0]

    between_boxes = []
    for b in boxes:
        y_top, y_bottom = b['bbox'][1], b['bbox'][3]
        x_left = b['bbox'][0]
        text = b['text'].strip()
        conf = b['ocr_confidence']

        if y_bottom <= y_first_bottom or y_top >= y_second_top:
            continue

        if x_left >= x_second_left:
            continue

        if b == second_date:
            continue

        if y_bottom <= y_second_top and y_bottom > y_second_top - 0.05:
            text_norm = normalize_text_pc(text)
            if any(normalize_text_pc(v) in text_norm for v in variants):
                continue

        if len(text) <= 2:
            continue

        text_norm = normalize_text_pc(text)
        if any(normalize_text_pc(v) in text_norm for v in variants):
            continue

        between_boxes.append(b)

    high_conf_boxes = [b for b in between_boxes if b['ocr_confidence'] > 0.86]

    if len(high_conf_boxes) >= 2:
        high_conf_boxes = [b for b in high_conf_boxes if b.get('classification', '').lower() != 'arabic']

    if len(high_conf_boxes) == 0:
        candidate_boxes = between_boxes
    elif len(high_conf_boxes) == 1:
        candidate_boxes = high_conf_boxes
    else:
        candidate_boxes = high_conf_boxes

    if len(candidate_boxes) == 1:
        selected_text = candidate_boxes[0]['text']
    elif len(candidate_boxes) > 1:
        candidate_boxes = sorted(candidate_boxes, key=lambda b: b['ocr_confidence'], reverse=True)
        selected_text = candidate_boxes[0]['text']
    else:
        selected_text = None

    return selected_text

def remove_accents_pc(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def extract_lieu_delivrance_pc(blocks):
    variants = [
        "délivré à", "delivre a", "delivré a", "délivre a", "dellivre a",
        "delivreà", "delivrea", "delivréà", "délivréà",
        "delivré a:", "délivré à:", "délivré à .", "délivré à :",
    ]

    # 1. First try exact match ignoring case (with accents)
    for block in blocks:
        original_text = block.get('text', '').strip()
        lower_text = original_text.lower()
        for variant in variants:
            if variant.lower() in lower_text:
                idx = lower_text.find(variant.lower())
                length = len(variant)
                after = original_text[idx + length:].strip(" :.")
                if after:
                    return after

    # 2. If no match found, try ignoring accents
    variants_no_accents = [remove_accents_pc(v).lower() for v in variants]

    for block in blocks:
        original_text = block.get('text', '').strip()
        text_no_accents = remove_accents_pc(original_text).lower()

        for variant_no_accents in variants_no_accents:
            if variant_no_accents in text_no_accents:
                idx = text_no_accents.find(variant_no_accents)
                length = len(variant_no_accents)
                after = original_text[idx + length:].strip(" :.")
                if after:
                    return after

    return None

def extract_license_number_pc(blocks):
    pattern = re.compile(r'\b\d{2}/\d{6}\b')

    for block in blocks:
        text = block.get('text', '')
        match = pattern.search(text)
        if match:
            return match.group(0)

    return None

def extract_cin_from_blocks_pc(blocks):
    pattern = re.compile(r'\b[A-Z]{1,3}\d{4,10}\b')
    results = []
    for block in blocks:
        text = block.get('text', '').upper()
        matches = pattern.findall(text)
        results.extend(matches)
    return results

def extract_dates_and_categories_pc(blocks):
    date_pattern = re.compile(r'\b(\d{2})[\s./-](\d{2})[\s./-](\d{4})\b')
    categories_list = {"AM", "A1", "A", "B", "C", "D", "EB", "EC", "ED"}

    result = {
        "date_de_naissance": None,
        "Delivre_le": None,
        "categories": []
    }

    # Filter blocks with decent OCR confidence
    filtered_blocks = [
        (block['text'], block['bbox'][1])
        for block in blocks
        if block.get('ocr_confidence', 0) > 0.5
    ]

    # Extract dates with y-position
    dates_with_positions = []
    for text, y in filtered_blocks:
        for match in date_pattern.finditer(text):
            dates_with_positions.append((match.group(0), y))
    dates_with_positions.sort(key=lambda x: x[1])  # sort top to bottom

    # Assign dates
    if len(dates_with_positions) >= 2:
        result["date_de_naissance"] = dates_with_positions[0][0]
        result["Delivre_le"] = dates_with_positions[1][0]
        delivre_le_y = dates_with_positions[1][1]
    elif len(dates_with_positions) == 1:
        result["Delivre_le"] = dates_with_positions[0][0]
        delivre_le_y = dates_with_positions[0][1]
    else:
        delivre_le_y = None

    # Extract categories from left-side blocks only (after Delivre_le)
    if delivre_le_y is not None:
        margin = 0.01
        left_below_blocks = [
            block for block in blocks
            if (
                block['bbox'][1] > delivre_le_y + margin and
                block.get('ocr_confidence', 0) > 0.6 and
                ((block['bbox'][0] + block['bbox'][2]) / 2) < 0.5  # left side
            )
        ]

        for block in left_below_blocks:
            words = re.findall(r'\b\w+\b', block['text'].upper())
            for w in words:
                if w in categories_list and w not in result["categories"]:
                    result["categories"].append(w)

    return result

def extract_permis_info(filtered_texts, ocr_results):
    result = {
        "pays": "Maroc",
        "type_de_carte": "PERMIS DE CONDUIRE",
        "prenom": None,
        "nom": None,
        "lieu_naissance": None,
        "lieu_delivrance": None,
        "numero_permis": None,
        "date_de_naissance": None,
        "delivre_le": None,
        "categories": [],
        "CIN": None
    }

    # Extract fields from OCR blocks
    result["prenom"] = extract_prenom_pc(ocr_results, filtered_texts)
    result["nom"] = extract_nom_from_blocks_pc(ocr_results)

    # Extract locations
    result["lieu_naissance"] = extract_lieu_de_naissance_pc(filtered_texts)
    result["lieu_delivrance"] = extract_lieu_delivrance_pc(filtered_texts)

    # Extract license number
    result["numero_permis"] = extract_license_number_pc(filtered_texts)

    # Extract dates and categories
    date_info = extract_dates_and_categories_pc(filtered_texts)
    result["date_de_naissance"] = date_info.get("date_de_naissance")
    result["delivre_le"] = date_info.get("Delivre_le")
    result["categories"] = date_info.get("categories", [])

    # Extract CIN
    CIN = extract_cin_from_blocks_pc(filtered_texts)
    result["CIN"] = CIN

    return result