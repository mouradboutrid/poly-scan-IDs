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

def extract_4_info_idb(blocks):
    cin_pattern = re.compile(r'[A-Z]{1,3}\d{4,10}')
    code_pattern = re.compile(r'^[A-Z0-9]{7,10}$')
    num_etat_pattern = re.compile(r'\b\d+(?:/\d+)+\b')

    cin_candidates = []
    code_candidates = []

    for block in blocks:
        raw_text = block.get('text', '').upper()
        bbox = block.get('bbox', [])
        if not bbox or len(bbox) != 4:
            continue

        cin_matches = cin_pattern.findall(raw_text)
        for cin_match in cin_matches:
            if len(cin_match) < 9:
                cin_candidates.append((bbox, cin_match))

        text_stripped = raw_text.strip()
        if len(text_stripped) < 11 and code_pattern.fullmatch(text_stripped):
            code_candidates.append((bbox, text_stripped))

    if cin_candidates:
        cin_candidates.sort(key=lambda item: (item[0][1], item[0][0]))
        cin_result = [cin_candidates[0][1]]
        cin_bbox = cin_candidates[0][0]
    else:
        cin_result = []
        cin_bbox = None

    if code_candidates:
        code_candidates.sort(key=lambda item: (item[0][1], -item[0][0]))
        code_result = [code_candidates[-1][1]]
        code_bbox = code_candidates[-1][0]
    else:
        code_result = []
        code_bbox = None

    num_etat_civil_results = []
    if cin_bbox and code_bbox:
        left_x = cin_bbox[2]
        right_x = code_bbox[0]
        max_y = max(cin_bbox[1], code_bbox[1])

        candidates = []
        for block in blocks:
            bbox = block.get('bbox', [])
            if not bbox or len(bbox) != 4:
                continue
            x_min, y_min, _, _ = bbox
            if left_x <= x_min <= right_x and y_min <= max_y + 0.05:
                text = block.get('text', '').strip()
                matches = num_etat_pattern.findall(text)
                for match in matches:
                    candidates.append((bbox, match))

        if candidates:
            if len(candidates) == 1:
                num_etat_civil_results = [candidates[0][1]]
            else:
                center_x = (left_x + right_x) / 2
                center_y = max_y / 2

                def distance_to_center(bbox):
                    x_min, y_min, x_max, y_max = bbox
                    box_center_x = (x_min + x_max) / 2
                    box_center_y = (y_min + y_max) / 2
                    return ((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2) ** 0.5

                candidates.sort(key=lambda c: distance_to_center(c[0]))
                num_etat_civil_results = [candidates[0][1]]

    for block in blocks:
        text = block.get('text', '').strip().upper()
        if re.fullmatch(r'SEXE\s*M', text):
            gender = 'M'
            break
        if re.fullmatch(r'SEXE\s*F', text):
            gender = 'F'
            break
    else:
        gender = None

    if gender is None and cin_bbox:
        cin_x_min, cin_y_min, cin_x_max, cin_y_max = cin_bbox
        for block in blocks:
            bbox = block.get('bbox', [])
            if not bbox or len(bbox) != 4:
                continue
            conf = block.get('conf', 0)
            if conf <= 0.8:
                continue
            x_min, y_min, x_max, y_max = bbox
            text = block.get('text', '').strip().upper()

            horizontally_aligned = (cin_x_min <= x_min <= cin_x_max) or (cin_x_min <= x_max <= cin_x_max)
            vertically_below = y_min > cin_y_max
            if horizontally_aligned and vertically_below:
                if text.startswith('FILE DE') or text.startswith('FILEDE'):
                    gender = 'F'
                    break
                if text.startswith('FILS DE') or text.startswith('FILSDE'):
                    gender = 'Male'
                    break

    if gender is None and code_bbox:
        code_x_min, code_y_min, code_x_max, code_y_max = code_bbox
        for block in blocks:
            bbox = block.get('bbox', [])
            if not bbox or len(bbox) != 4:
                continue
            x_min, y_min, x_max, y_max = bbox
            text = block.get('text', '').strip().upper()
            horizontally_aligned = (code_x_min <= x_min <= code_x_max) or (code_x_min <= x_max <= code_x_max)
            vertically_below = y_min > code_y_max
            if horizontally_aligned and vertically_below and len(text) <= 6:
                if re.search(r'\bSEXE\s*M\b', text):
                    gender = 'M'
                    break
                if re.search(r'\bSEXE\s*F\b', text):
                    gender = 'F'
                    break

        if gender is None:
            for block in blocks:
                bbox = block.get('bbox', [])
                if not bbox or len(bbox) != 4:
                    continue
                x_min, y_min, x_max, y_max = bbox
                text = block.get('text', '').strip().upper()
                horizontally_aligned = (code_x_min <= x_min <= code_x_max) or (code_x_min <= x_max <= code_x_max)
                vertically_below = y_min > code_y_max
                if horizontally_aligned and vertically_below and len(text) <= 6:
                    if re.search(r'\bM\b', text):
                        gender = 'Male'
                        break
                    if re.search(r'\bF\b', text):
                        gender = 'Female'
                        break

    return cin_result, code_result, num_etat_civil_results, gender

def extract_address_from_blocks_idb(blocks):
    address_keywords = {'ADRESSE', 'ADRESS', 'ADRES', 'ADESSE'}

    # First pass: Look for "adresse-like" word in the first two words of the block
    for block in blocks:
        text = block.get('text', '').upper().strip()
        words = re.split(r'\s+', text)
        first_two = words[:2]
        if any(word in address_keywords for word in first_two):
            return text

    # Fallback: Positional logic using CIN
    cin_pattern = re.compile(r'[A-Z]{1,3}\d{4,10}')
    cin_candidates = []

    for block in blocks:
        text = block.get('text', '').upper()
        bbox = block.get('bbox', [])
        if len(bbox) != 4:
            continue
        cin_matches = cin_pattern.findall(text)
        for match in cin_matches:
            cin_candidates.append((bbox, match))

    if not cin_candidates:
        return None

    # Get top-left-most CIN block
    cin_candidates.sort(key=lambda item: (item[0][1], item[0][0]))
    cin_bbox = cin_candidates[0][0]
    cin_x_min, _, _, cin_y_max = cin_bbox

    # Find blocks that are below CIN and roughly horizontally aligned
    aligned_blocks = []
    for block in blocks:
        bbox = block.get('bbox', [])
        if len(bbox) != 4:
            continue
        x_min, y_min, _, _ = bbox
        if abs(x_min - cin_x_min) < 0.1 and y_min > cin_y_max:
            aligned_blocks.append((y_min, block))

    # Sort by vertical position and take the third block (index 2)
    aligned_blocks.sort(key=lambda item: item[0])
    if len(aligned_blocks) >= 3:
        return aligned_blocks[2][1].get('text', '').strip()

    return None

def extract_nom_de_pere_mere_idb(blocks, cin_text):
    pere_keywords = ["FILS DE", "FILSDE", "FILEDE", "FILE DE"]
    mere_keywords = ["ET DE", "ETDE"]

    def starts_with_keyword(text, keywords):
        text_upper = text.upper().strip()
        return any(text_upper.startswith(k) for k in keywords)

    # Step 1: Find CIN block
    cin_text_upper = cin_text.upper()
    cin_block = next((b for b in blocks if cin_text_upper in b.get("text", "").upper()), None)

    cin_bbox = cin_block["bbox"]
    x_center = (cin_bbox[0] + cin_bbox[2]) / 2
    y_start = cin_bbox[3]

    # Step 2: Find blocks with high confidence and matching keywords
    pere_block = None
    mere_block = None
    for block in blocks:
        if block.get("ocr_confidence", 0) < 0.75:
            continue
        text = block.get("text", "").strip()
        if starts_with_keyword(text, pere_keywords) and not pere_block:
            pere_block = block
        elif starts_with_keyword(text, mere_keywords) and not mere_block:
            mere_block = block

    if pere_block and mere_block:
        return pere_block["text"], mere_block["text"]


    if not cin_block or "bbox" not in cin_block:
        return None, None
    # Step 3: Fallback using vertical line
    blocks_below = [
        b for b in blocks
        if b.get("bbox") and b["bbox"][1] > y_start
    ]
    blocks_below.sort(key=lambda b: b["bbox"][1])

    vertical_blocks = [
        b for b in blocks_below
        if b["bbox"][0] <= x_center <= b["bbox"][2]
    ]

    selected_blocks = vertical_blocks[:2]

    # Step 4: Use hybrid logic
    if pere_block or mere_block:
        if pere_block and not mere_block and len(selected_blocks) >= 2:
            return pere_block["text"], selected_blocks[1]["text"]
        elif mere_block and not pere_block and len(selected_blocks) >= 2:
            return selected_blocks[0]["text"], mere_block["text"]
        elif pere_block and not mere_block and len(selected_blocks) == 1:
            return pere_block["text"], None
        elif mere_block and not pere_block and len(selected_blocks) == 1:
            return None, mere_block["text"]

    # Step 5: If no keyword blocks at all, use both from vertical line
    if len(selected_blocks) >= 2:
        return selected_blocks[0]["text"], selected_blocks[1]["text"]
    else:
        return None, None


def extract_idback_info(blocks):

    cin_result, code_result, num_etat_civil_results, gender = extract_4_info_idb(blocks)
    adresse = extract_address_from_blocks_idb(blocks)
    try:
      nom_de_pere, nom_de_mere = extract_nom_de_pere_mere_idb(blocks, cin_result[0])
    except :
      nom_de_pere, nom_de_mere = None, None

    result = {
        "pays": "Maroc",
        "type_de_carte": "Carte Nationale d'Identité",
        "CIN": (cin_result[0] if cin_result else None),
        "Code": (code_result[0] if code_result else None),
        "Num_Civil": (num_etat_civil_results[0] if num_etat_civil_results else None),
        "Sexe": gender,
        "Pere": nom_de_pere,
        "Mere": nom_de_mere, 
        "Adresse": adresse
    }

    return result