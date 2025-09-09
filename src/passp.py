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

def extract_passport_dates(blocks):
    date_pattern = re.compile(r'\b(\d{2})[/-](\d{2})[/-](\d{4})\b')

    date_blocks = []

    for block in blocks:
        if block.get("ocr_confidence", 0) < 0.4:
            continue

        text = block["text"]
        for match in date_pattern.finditer(text):
            date_str = match.group(0)
            day, month, year = match.groups()
            y_position = block["bbox"][1]
            date_blocks.append({
                "text": date_str,
                "year": int(year),
                "y": y_position
            })

    # Sort top-to-bottom by y-position
    date_blocks.sort(key=lambda x: x["y"])

    result = {
        "date_de_naissance": None,
        "date_de_delivrance": None,
        "date_dexpiration": None
    }

    # Try to find the first valid trio in chronological order
    for i in range(len(date_blocks) - 2):
        d1, d2, d3 = date_blocks[i], date_blocks[i+1], date_blocks[i+2]
        if d1["year"] < d2["year"] < d3["year"]:
            result["date_de_naissance"] = d1["text"]
            result["date_de_delivrance"] = d2["text"]
            result["date_dexpiration"] = d3["text"]
            return result

    return result


def extract_authorite(blocks, date_delivrance_text, y_tolerance=0.02):
    date_block = None
    for block in blocks:
        if block.get("ocr_confidence", 0) < 0.6:
            continue
        if date_delivrance_text in block["text"]:
            date_block = block
            break
    if not date_block:
        return None
    date_y = date_block["bbox"][1]
    date_x_max = date_block["bbox"][2]

    candidates = []
    for block in blocks:
        if block.get("ocr_confidence", 0) < 0.6:
            continue
        bbox = block["bbox"]
        block_y_top = bbox[1]
        block_y_bottom = bbox[3]
        if abs(block_y_top - date_y) < y_tolerance or abs(block_y_bottom - date_y) < y_tolerance:
            if bbox[0] > date_x_max:
                candidates.append(block)

    if not candidates:
        return None
    candidates.sort(key=lambda b: b["bbox"][0])
    return candidates[0]["text"]


def extract_passport_sexe(blocks, birth_date):
    if not birth_date:
        return None

    target_block = None
    margin_y = 0.01  # Vertical tolerance for alignment

    # Step 1: Find the block containing the birth date
    for block in blocks:
        if birth_date in block.get("text", ""):
            target_block = block
            break

    if not target_block:
        return None

    birth_y_min = target_block["bbox"][1]
    birth_y_max = target_block["bbox"][3]
    birth_x_min = target_block["bbox"][0]

    # Step 2: Scan all blocks to the left on the same line
    candidates = []
    for block in blocks:
        text = block.get("text", "").strip().upper()
        x_min, y_min, x_max, y_max = block["bbox"]

        # Must be aligned with the birth date row
        if y_min >= birth_y_min - margin_y and y_max <= birth_y_max + margin_y:
            # Must be to the left of birth date block
            if x_max <= birth_x_min:
                # Allow max 3 characters with M or F inside
                if len(text) <= 3 and any(letter in text for letter in ("M", "F")):
                    candidates.append((x_min, text))

    # Step 3: Return the closest valid gender marker to the birth date
    if candidates:
        candidates.sort(reverse=True)  # rightmost valid candidate first
        for _, text in candidates:
            if "M" in text:
                return "M"
            if "F" in text:
                return "F"

    return None

def extract_cin_and_passport_number(blocks):
    passport_regex = re.compile(r'\b[A-Z]{2}\d{7}\b')
    cin_regex = re.compile(r'\b[A-Z]{1,2}\d{4,7}\b')    #

    passport_candidates = []
    cin_candidates = []

    for block in blocks:
        text = block.get("text", "").strip().upper()
        if not text:
            continue

        x_min, y_min, x_max, y_max = block["bbox"]

        # Check for passport number (top right region)
        if passport_regex.fullmatch(text) and x_min >= 0.5 and y_min <= 0.4:
            passport_candidates.append((y_min, block))

        # Check for CIN (bottom left region)
        if cin_regex.fullmatch(text) and x_max <= 0.5 and y_max >= 0.6:
            cin_candidates.append((y_max, block))

    result = {
        "numero_passport": passport_candidates[0][1]["text"] if passport_candidates else None,
        "cin": cin_candidates[0][1]["text"] if cin_candidates else None
    }

    return result


def extract_passport_type(blocks, passport_number):
    passport_types = {"P", "PP", "PD", "PS", "S", "PL"}

    passport_number = passport_number.upper()
    target_block = None

    for block in blocks:
        text = block.get("text", "").strip().upper()
        if text == passport_number:
            target_block = block
            break

    if not target_block:
        return None

    y_center = (target_block["bbox"][1] + target_block["bbox"][3]) / 2
    x_min = target_block["bbox"][0]

    candidates = []

    for block in blocks:
        if block == target_block:
            continue
        y_block_center = (block["bbox"][1] + block["bbox"][3]) / 2
        x_max = block["bbox"][2]

        if abs(y_block_center - y_center) < 0.02:
            if x_max < x_min:
                text = block.get("text", "").strip().upper()
                if text in passport_types:
                    candidates.append((x_max, text))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # Fallback: check last two lines of blocks for a starting string matching passport types
    blocks_sorted = sorted(blocks, key=lambda b: b["bbox"][1], reverse=True)
    last_two_lines = blocks_sorted[:2]

    for block in last_two_lines:
        text = block.get("text", "").strip().upper()
        if not text:
            continue
        prefix = text[:2]
        if prefix[0] == "P" and prefix not in passport_types:
            return "P"
        if prefix in passport_types:
            return prefix

    return None

import unicodedata

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').lower()

def extract_nom_passport(blocks, numero_passport=None):
    def find_blocks_containing(substring, accent_sensitive=True):
        results = []
        search_str = substring.lower() if accent_sensitive else normalize_text(substring)
        for block in blocks:
            text = block.get("text", "")
            target_text = text if accent_sensitive else normalize_text(text)
            if search_str in target_text:
                results.append(block)
        return results

    def left_side_blocks_below(target_block, count=2):
        y_max = target_block["bbox"][3]
        candidates = []
        for block in blocks:
            bx_min, by_min, bx_max, by_max = block["bbox"]
            if by_min > y_max and bx_max <= 0.5:
                candidates.append(block)
        candidates.sort(key=lambda b: b["bbox"][1])
        return candidates[:count]

    prenoms_blocks = [b for b in blocks if b.get("text","").lower().startswith("prénoms")]
    if prenoms_blocks:
        block = prenoms_blocks[0]
        above_candidates = [b for b in blocks if abs(b["bbox"][1] - block["bbox"][1]) < 0.05 and b["bbox"][3] < block["bbox"][1] and b["bbox"][0] < 0.5]
        if above_candidates:
            above_candidates.sort(key=lambda b: b["bbox"][1], reverse=True)
            return above_candidates[0]["text"]

    prenoms_blocks = find_blocks_containing("prenoms", accent_sensitive=False)
    if prenoms_blocks:
        block = prenoms_blocks[0]
        above_candidates = [b for b in blocks if abs(b["bbox"][1] - block["bbox"][1]) < 0.05 and b["bbox"][3] < block["bbox"][1] and b["bbox"][0] < 0.5]
        if above_candidates:
            above_candidates.sort(key=lambda b: b["bbox"][1], reverse=True)
            return above_candidates[0]["text"]

    given_blocks = []
    for b in blocks:
        text_norm = normalize_text(b.get("text",""))
        if "given" in text_norm:
            given_blocks.append(b)
    if given_blocks:
        block = given_blocks[0]
        above_candidates = [b for b in blocks if abs(b["bbox"][1] - block["bbox"][1]) < 0.05 and b["bbox"][3] < block["bbox"][1] and b["bbox"][0] < 0.5]
        if above_candidates:
            above_candidates.sort(key=lambda b: b["bbox"][1], reverse=True)
            return above_candidates[0]["text"]

    nom_blocks = []
    for b in blocks:
        text = b.get("text","").lower()
        if text.startswith("nom") or text.startswith("nom/") or text.startswith("noml") or text.startswith("nom/name"):
            nom_blocks.append(b)
    if nom_blocks:
        block = nom_blocks[0]
        below_candidates = left_side_blocks_below(block, count=2)
        if not below_candidates:
            return None
        if len(below_candidates) == 1:
            return below_candidates[0]["text"]
        c1, c2 = below_candidates[0], below_candidates[1]
        c1_conf = c1.get("ocr_confidence",0)
        c2_conf = c2.get("ocr_confidence",0)
        if c1_conf > 0.86 and c2_conf > 0.86:
            c1_class = c1.get("classification","").lower()
            c2_class = c2.get("classification","").lower()
            if c1_class == "french" and c2_class == "french":
                return c2["text"]
            elif c1_class == "french":
                return c1["text"]
            elif c2_class == "french":
                return c2["text"]
            else:
                return c1["text"]
        elif c1_conf > 0.86:
            return c1["text"]
        elif c2_conf > 0.86:
            return c2["text"]
        else:
            return c1["text"]

    if numero_passport:
        target_block = None
        for b in blocks:
            if b.get("text","").strip().upper() == numero_passport.upper():
                target_block = b
                break
        if target_block:
            y_max = target_block["bbox"][3]
            candidates = []
            for b in blocks:
                bx_min, by_min, bx_max, by_max = b["bbox"]
                if by_min > y_max and bx_max <= 0.5:
                    candidates.append(b)
            candidates.sort(key=lambda b: b["bbox"][1])
            candidates = candidates[:3]
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]["text"]
            filtered = [c for c in candidates if c.get("ocr_confidence",0) > 0.86]
            if filtered:
                french_filtered = [c for c in filtered if c.get("classification","").lower() == "french"]
                if french_filtered:
                    return french_filtered[0]["text"]
                return filtered[0]["text"]
            return candidates[0]["text"]

    return None

def extract_prenom_passport(ocr_blocks, nom_value=None):

    # Helper function to find text with tolerance
    def find_text_pattern(pattern, blocks):
        results = []
        for block in blocks:
            text = block.get('text', '').upper()
            if re.search(pattern, text):
                results.append(block)
        return results

    # Improved spatial relationship function
    def get_boxes_in_relation(target_block, blocks, relation='above', x_tolerance=0.15, max_distance=0.3):
        target_bbox = target_block.get('bbox', (0, 0, 0, 0))
        target_center_x = (target_bbox[0] + target_bbox[2]) / 2

        candidates = []
        for block in blocks:
            if block == target_block:
                continue

            block_bbox = block.get('bbox', (0, 0, 0, 0))
            block_center_x = (block_bbox[0] + block_bbox[2]) / 2

            # Check if in same vertical column
            if abs(block_center_x - target_center_x) < x_tolerance:
                distance = None

                if relation == 'above' and block_bbox[3] < target_bbox[1]:
                    distance = target_bbox[1] - block_bbox[3]
                elif relation == 'below' and block_bbox[1] > target_bbox[3]:
                    distance = block_bbox[1] - target_bbox[3]

                if distance is not None and distance < max_distance:
                    candidates.append((block, distance))

        # Sort by distance and return
        return [block for block, dist in sorted(candidates, key=lambda x: x[1])]

    # TRY 1: Look for explicit "Prénoms" or "Given Names" label (MOST RELIABLE)
    prenoms_blocks = find_text_pattern(r'PRÉNOMS|PRENOMS|GIVEN.*NAMES', ocr_blocks)

    if prenoms_blocks:
        prenoms_block = prenoms_blocks[0]
        boxes_below = get_boxes_in_relation(prenoms_block, ocr_blocks, 'below')

        # Look for the actual given name below the label
        for box in boxes_below:
            text = box.get('text', '').strip()
            # More strict validation for given names
            if (2 <= len(text) <= 20 and
                re.match(r'^[A-Z][A-Za-z\.]*$', text) and
                not any(keyword in text.upper() for keyword in ['MAROC', 'MAROCAIN', 'PASSEPORT', 'NATIONAL', 'SEXE', 'DATE']) and
                not re.search(r'\d', text)):
                return text

    # TRY 2: Use nom value with BETTER filtering
    if nom_value:
        # Find the nom block
        nom_blocks = []
        clean_nom = re.sub(r'[^A-Z]', '', nom_value.upper())

        for block in ocr_blocks:
            clean_text = re.sub(r'[^A-Z]', '', block.get('text', '').upper())
            if clean_nom in clean_text and len(clean_text) >= len(clean_nom) * 0.8:
                nom_blocks.append(block)

        if nom_blocks:
            nom_block = nom_blocks[0]
            boxes_below = get_boxes_in_relation(nom_block, ocr_blocks, 'below')

            # Check if any box below looks like a given name with BETTER filtering
            for box in boxes_below:
                text = box.get('text', '').strip()

                # Skip nationality words and other non-name text
                if any(keyword in text.upper() for keyword in ['MAROCAIN', 'MAROCAINE', 'NATIONAL', 'SEXE', 'DATE']):
                    continue

                # Skip text with numbers or special characters (except dots for initials)
                if re.search(r'[^A-Za-z\.]', text) and not text.endswith('.'):
                    continue

                # Look for actual name text with reasonable length
                if (2 <= len(text) <= 20 and re.match(r'^[A-Z][A-Za-z\.]*$', text)):
                    return text

    # TRY 3: Find blocks containing "Nationalit" pattern
    nationalite_blocks = find_text_pattern(r'NATIONA(LIT|LITY|TÉ|TY)', ocr_blocks)
    if nationalite_blocks:
        nationalite_block = nationalite_blocks[0]
        boxes_above = get_boxes_in_relation(nationalite_block, ocr_blocks, 'above')

        # Look for boxes that are NOT the surname and look like given names
        for box in boxes_above:
            text = box.get('text', '').strip()

            # Skip if this is likely the surname (matches provided nom_value)
            if nom_value and nom_value.upper() in text.upper():
                continue

            # Skip label boxes and nationality words
            if any(keyword in text.upper() for keyword in ['PRÉNOMS', 'PRENOMS', 'GIVEN', 'NOM', 'NAME', 'MAROCAIN']):
                continue

            # Skip non-name text
            if re.match(r'^[^A-Za-z]*$', text) or re.search(r'\d', text):
                continue

            # This should be the given name!
            if 2 <= len(text) <= 20:
                return text

    # TRY 4: Direct search for name patterns (fallback)
    for block in ocr_blocks:
        text = block.get('text', '').strip()
        # Look for text that matches name patterns but excludes known non-name words
        if (2 <= len(text) <= 20 and
            re.match(r'^[A-Z][A-Za-z\.]*$', text) and
            not any(keyword in text.upper() for keyword in ['MAROC', 'PASSEPORT', 'NOM', 'PRÉNOMS', 'NATIONAL', 'SEXE', 'DATE', 'CARD']) and
            (not nom_value or nom_value.upper() not in text.upper())):
            return text

    return None

def get_vertical_strip_between_dates(ocr_data, date1, date2, overlap_margin=0.001):

    def find_bbox_by_text(target_text):
        for entry in ocr_data:
            if target_text.strip() in entry['text']:
                return entry['bbox']
        return None

    bbox1 = find_bbox_by_text(date1)
    bbox2 = find_bbox_by_text(date2)

    if not bbox1 or not bbox2:
        raise ValueError(f"One or both dates not found: '{date1}', '{date2}'")

    y1_top, y1_bottom = bbox1[1], bbox1[3]
    y2_top, y2_bottom = bbox2[1], bbox2[3]

    # Define the vertical range between dates
    y_top = min(y1_bottom, y2_bottom)
    y_bottom = max(y1_top, y2_top)

    def overlaps(a_top, a_bottom, b_top, b_bottom, margin=0.001):
        return not (a_bottom < b_top + margin or a_top > b_bottom - margin)

    # Filter entries that are within vertical range and NOT overlapping date lines
    filtered = []
    for entry in ocr_data:
        e_top, e_bottom = entry['bbox'][1], entry['bbox'][3]

        if y_top < e_top and e_bottom < y_bottom:
            # Fully between the dates
            filtered.append(entry)
        else:
            if overlaps(e_top, e_bottom, y1_top, y1_bottom, overlap_margin):
                continue
            if overlaps(e_top, e_bottom, y2_top, y2_bottom, overlap_margin):
                continue
            if y_top < e_top and e_bottom < y_bottom:
                filtered.append(entry)

    return filtered

def split_blocks_by_left_right(blocks):

    # Calculate x-centers for all blocks
    x_centers = []
    for block in blocks:
        x_min, _, x_max, _ = block['bbox']
        x_center = (x_min + x_max) / 2
        x_centers.append(x_center)

    # Compute the median x_center as the vertical split
    median_x = sorted(x_centers)[len(x_centers) // 2]

    left_blocks = []
    right_blocks = []

    for block, x_center in zip(blocks, x_centers):
        if x_center < median_x:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    return left_blocks, right_blocks

def extract_closest_field(texts):
    label_keywords = ['lieu', 'birth', 'domicile', 'date', 'naissanca', 'place', 'délivr']
    fallback_keyword = 'maroc'

    def is_close_word(word, keywords, max_dist=2):
        """Return True if word is within max_dist edit distance of any keyword"""
        return any(difflib.SequenceMatcher(None, word, kw).ratio() >= (1 - max_dist / max(len(word), len(kw))) for kw in keywords)

    # Only one box
    if len(texts) == 1:
        entry = texts[0]
        if entry.get('classification', '').lower() == 'french' or entry.get('ocr_confidence', 0) > 0.5:
            return [entry['text']]
        else:
            return None

    matched_labels = []

    # Try to find label using exact keyword match
    for entry in texts:
        text = entry['text'].lower()
        if any(kw in text for kw in label_keywords):
            matched_labels.append(entry)

    results = []

    # Use closest box to label(s)
    for label in matched_labels:
        lx_min, ly_min, lx_max, ly_max = label['bbox']
        label_center = ((lx_min + lx_max) / 2, (ly_min + ly_max) / 2)

        closest_candidate = None
        closest_distance = float('inf')

        for candidate in texts:
            if candidate == label:
                continue

            cx_min, cy_min, cx_max, cy_max = candidate['bbox']
            candidate_center = ((cx_min + cx_max) / 2, (cy_min + cy_max) / 2)

            distance = np.linalg.norm(np.array(candidate_center) - np.array(label_center))

            if distance < closest_distance:
                closest_distance = distance
                closest_candidate = candidate

        if closest_candidate:
            results.append(closest_candidate['text'])

    # Fallback – look for "maroc" in any text
    if not results:
        for entry in texts:
            if fallback_keyword in entry['text'].lower():
                results.append(entry['text'])
                break

    # Fuzzy label detection for 2-box edge case
    if not results and len(texts) == 2:
        for entry in texts:
            text_words = entry['text'].lower().split()
            for word in text_words:
                if is_close_word(word, label_keywords, max_dist=2):
                    # Return the *other* entry as the candidate
                    other = [e for e in texts if e != entry]
                    if other:
                        return [other[0]['text']]

    return results if results else None


def looks_like_address(text):
    """
    Return True if text contains symbols typically found in address blocks.
    """
    symbols = set('@[{!#%*]')
    return any(char in text for char in symbols)

def fuzzy_match_in_text(text, keywords, max_distance=2):
    """
    Check if any keyword approximately appears inside the text allowing max_distance errors.
    """
    text = text.lower()
    for keyword in keywords:
        keyword = keyword.lower()
        # Slide over the text with a window equal to keyword length and check fuzzy similarity
        length = len(keyword)
        for i in range(len(text) - length + 1):
            window = text[i:i+length]
            # Calculate Levenshtein-like distance using difflib SequenceMatcher ratio
            seq = difflib.SequenceMatcher(None, keyword, window)
            similarity = seq.ratio()
            # similarity of 1.0 means exact match; allow some fuzziness, e.g. >= 0.7
            if similarity >= 0.7:
                # Allow some differences in length by checking actual edit distance as well
                # But difflib ratio should be enough here
                return True
    return False

def extract_address(blocks, authorite_text):
    label_keywords = ['domicile', 'residence', 'authorit', 'authority', 'lieu', 'authori']

    # Locate and exclude the authorite block and anything below it
    authorite_box = None
    for block in blocks:
        if authorite_text and authorite_text.lower() in block['text'].lower():
            authorite_box = block
            break

    if authorite_box:
        _, a_ymin, _, _ = authorite_box['bbox']
        blocks = [b for b in blocks if b['bbox'][1] < a_ymin]

    if not blocks:
        return None

    # If only one block, return it if it looks like a valid address
    if len(blocks) == 1:
        b = blocks[0]
        if (
            b['classification'].lower() == 'french'
            or b['ocr_confidence'] > 0.5
            or looks_like_address(b['text'])
        ):
            return b['text']
        return None

    # Remove any label-like blocks using fuzzy matching on the full text
    filtered_blocks = []
    for b in blocks:
        if fuzzy_match_in_text(b['text'], label_keywords):
            # This block looks like a label, so skip it
            continue
        filtered_blocks.append(b)

    if not filtered_blocks:
        return None

    # Return cleaned result depending on how many blocks are left
    if len(filtered_blocks) == 1:
        return filtered_blocks[0]['text']
    elif len(filtered_blocks) == 2:
        return f"{filtered_blocks[0]['text']} {filtered_blocks[1]['text']}"
    else:
        # Sort left-to-right (x_min) if more than 2 blocks
        filtered_blocks.sort(key=lambda b: b['bbox'][0])
        return ' '.join(b['text'] for b in filtered_blocks)

def extract_passport_info(filtered_texts):
    # Extract all pieces
    cin_num = extract_cin_and_passport_number(filtered_texts) or {}
    dates = extract_passport_dates(filtered_texts) or {}
    authorite = extract_authorite(filtered_texts, dates.get('date_de_delivrance', '')) or ''
    sexe = extract_passport_sexe(filtered_texts, dates.get('date_de_naissance', '')) or ''
    passport_type = extract_passport_type(filtered_texts, cin_num.get('numero_passport', '')) or ''
    nom = extract_nom_passport(filtered_texts, cin_num.get('numero_passport', '')) or ''
    prenom = extract_prenom_passport(filtered_texts, nom_value=nom) or ''

    # Filter text vertically between dates, split into left/right blocks
    filt_text = get_vertical_strip_between_dates(filtered_texts, dates.get('date_de_naissance', ''), dates.get('date_de_delivrance', ''))
    left_blocks, right_blocks = split_blocks_by_left_right(filt_text)

    place_of_birth_list = extract_closest_field(left_blocks) or []
    place_of_birth = place_of_birth_list[0] if place_of_birth_list else ''

    address = extract_address(right_blocks, authorite) or ''

    # Compose flat dictionary
    result = {
        "pays": "Maroc",
        "type_de_carte": "PASSPORT",
        "Naionalite": "Marocaine",
        'numero_passport': cin_num.get('numero_passport', ''),
        'cin': cin_num.get('cin', ''),
        'date_de_naissance': dates.get('date_de_naissance', ''),
        'date_de_delivrance': dates.get('date_de_delivrance', ''),
        'date_dexpiration': dates.get('date_dexpiration', ''),
        'authorite': authorite,
        'sexe': sexe,
        'passport_type': passport_type,
        'nom': nom,
        'prenom': prenom,
        'place_of_birth': place_of_birth,
        'address': address,
    }

    return result