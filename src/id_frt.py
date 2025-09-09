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

def find_name_blocks_idf(blocks):
    target_words = {"carte", "nationale", "identite"}
    header_block = None

    for block in blocks:
        text = block.get('text', '').lower()
        if any(word in text for word in target_words):
            header_block = block
            break
    if not header_block:
        return None, None

    all_x_mins = [b['bbox'][0] for b in blocks]
    all_x_maxs = [b['bbox'][2] for b in blocks]
    min_x = min(all_x_mins)
    max_x = max(all_x_maxs)
    horizontal_mid = (min_x + max_x) / 2

    hx_min, hy_min, hx_max, hy_max = header_block['bbox']

    candidates = [b for b in blocks
                  if b['bbox'][1] > hy_max and
                  b['bbox'][0] < horizontal_mid]

    candidates.sort(key=lambda b: b['bbox'][1])

    if len(candidates) < 2:
        return None, None

    prenom = candidates[0]['text']
    nom = candidates[1]['text']

    return prenom, nom


def extract_cin_from_blocks_idf(blocks):
    pattern = re.compile(r'\b[A-Z]{1,3}\d{4,10}\b')
    results = []
    for block in blocks:
        text = block.get('text', '').upper()
        matches = pattern.findall(text)
        results.extend(matches)
    return results


def extract_dates_from_ocr_blocks_idf(blocks):
    date_pattern = re.compile(r'\b(\d{2})[\s./-](\d{2})[\s./-](\d{4})\b')
    year_pattern = re.compile(r'\b(19\d{2})\b')

    result = {
        "date_de_naissance": None,
        "valable_jusqua": None
    }

    texts = [block['text'] for block in blocks]
    found_dates = []

    for text in texts:
        matches = date_pattern.findall(text)
        raw_matches = date_pattern.finditer(text)
        for match in raw_matches:
            full_raw_date = match.group(0)
            found_dates.append(full_raw_date)
        if len(found_dates) >= 2:
            break

    if len(found_dates) >= 2:
        result["date_de_naissance"] = found_dates[0]
        result["valable_jusqua"] = found_dates[1]
        return result

    if len(found_dates) == 1:
        result["valable_jusqua"] = found_dates[0]
        for text in texts:
            clean_text = re.sub(r'[^\w\s]', '', text)
            year_matches = year_pattern.findall(clean_text)
            for y in year_matches:
                if y != found_dates[0][-4:]:
                    result["date_de_naissance"] = y
                    return result
        return result

    for text in texts:
        clean_text = re.sub(r'[^\w\s]', '', text)
        year_matches = year_pattern.findall(clean_text)
        if year_matches:
            result["date_de_naissance"] = year_matches[0]
            break

    return result


def normalize_text_idf(text):
    if not text:
        return ''
    return re.sub(r'[^a-z]', '', text.lower())


def similarity_ratio_idf(a, b):
    a_norm = normalize_text_idf(a)
    b_norm = normalize_text_idf(b)
    if not a_norm or not b_norm:
        return 0
    matches = sum(1 for x, y in zip(a_norm, b_norm) if x == y)
    return matches / max(len(a_norm), len(b_norm))


def convert_date_to_iso_idf(date_str):
    date_pattern = re.compile(r'(\d{2})[\s./-](\d{2})[\s./-](\d{4})')
    m = date_pattern.search(date_str)
    if m:
        day, month, year = m.groups()
        try:
            return datetime.strptime(f'{day}-{month}-{year}', '%d-%m-%Y').date().isoformat()
        except:
            return None
    return None


def clean_ocr_blocks_idf(blocks):
    unwanted_texts = ['royaume du maroc', 'royaume maroc', 'carte nationale didentite',
                      "carte nationale d'identite", 'carte nationale identite', 'carte national didentite',
                      "carte national d'identite", 'carte national identite', 'specmen', 'specimen', 'ne le',
                      'née le', 'neé le', 'nee le', 'néle', 'néé le', 'nééle', 'neéle', 'du maroc', 'royaume du',
                      'nationale didentite', "nationale d'identite", 'nationale identite', 'national didentite',
                      "national d'identite", 'national identite', 'carte didentite', "carte d'identite", 'carte identite',
                      "maroc", "auime du maroc", "N\"", 'Valable jusqu\'au', "royaume", "ROYAUUE DU MADOO"
                      ]

    dates = extract_dates_from_ocr_blocks_idf(blocks)
    dates_to_remove = {v for v in dates.values() if v is not None}

    cin_list = extract_cin_from_blocks_idf(blocks)
    cin_set = set(cin_list)

    cleaned_blocks = []
    for block in blocks:
        text = block.get('text', '').strip()
        if len(text) <= 1:
            continue

        if any(similarity_ratio_idf(text, unwanted) >= 0.9 for unwanted in unwanted_texts):
            continue

        date_substrings = re.findall(r'\b\d{2}[\s./-]\d{2}[\s./-]\d{4}\b', text)
        block_dates = set()
        for ds in date_substrings:
            iso_date = convert_date_to_iso_idf(ds)
            if iso_date:
                block_dates.add(iso_date)

        if block_dates.intersection(dates_to_remove):
            continue

        text_upper = text.upper()
        if any(cin in text_upper for cin in cin_set):
            continue

        cleaned_blocks.append(block)

    return cleaned_blocks


def extract_names_with_validation_idf(blocks):
    cleaned_blocks = clean_ocr_blocks_idf(blocks)
    prenom, nom = find_name_blocks_idf(blocks)

    if prenom is None or nom is None:
        return None, None

    cleaned_texts = [block['text'].lower() for block in cleaned_blocks]

    def text_present(text):
        norm_text = normalize_text_idf(text)
        return any(norm_text in normalize_text_idf(ct) for ct in cleaned_texts)

    if not text_present(prenom) or not text_present(nom):
        return None, None

    return prenom, nom


def remove_name_blocks_idf(blocks, prenom, nom):
    prenom_norm = normalize_text_idf(prenom)
    nom_norm = normalize_text_idf(nom)

    filtered_blocks = []
    for block in blocks:
        text_norm = normalize_text_idf(block.get('text', ''))
        if text_norm == prenom_norm or text_norm == nom_norm:
            continue
        filtered_blocks.append(block)

    return filtered_blocks


def extract_lieu_idf(raw_blocks, clean_blocks):
    dates = extract_dates_from_ocr_blocks_idf(raw_blocks)
    ref_block = None

    for block in raw_blocks:
        text_lower = block['text'].lower()
        if any(phrase in text_lower for phrase in ['neele', 'nele', 'ne le', 'né le', 'née le', 'ne lé', 'né lé', 'née lé', 'ne l', 'né l', 'née l', 'ne le.', 'né le.', 'née le.', 'ne-le', 'né-le', 'née-le', 'nè le', 'nè lé', 'nè-lé', 'nè-le', 'ne lè', 'né lè', 'née lè']):
            ref_block = block
            break

    if not ref_block and dates['date_de_naissance']:
        date_str = dates['date_de_naissance'].replace('-', '.')
        for block in raw_blocks:
            if date_str in block['text']:
                ref_block = block
                break

    if not ref_block:
        return None

    ref_x_min, ref_y_min, ref_x_max, ref_y_max = ref_block['bbox']

    candidates = [b for b in clean_blocks
                  if b['bbox'][0] <= ref_x_min + 0.05
                  and b['bbox'][1] > ref_y_max]

    if not candidates:
        return None

    candidates.sort(key=lambda b: b['bbox'][1])

    first_line = candidates[0]
    first_text = first_line['text']

    if len(candidates) > 1:
        second_line = candidates[1]

        vertical_gap = second_line['bbox'][1] - first_line['bbox'][3]
        horizontal_gap = abs(second_line['bbox'][0] - first_line['bbox'][0])

        if vertical_gap < 0.03 and horizontal_gap < 0.03 and len(first_text) > 5:
            return first_text + ' ' + second_line['text']

    return first_text


def extract_id_info(blocks):
    cin = extract_cin_from_blocks_idf(blocks)
    dates = extract_dates_from_ocr_blocks_idf(blocks)

    clean = clean_ocr_blocks_idf(blocks)
    prenom, nom = extract_names_with_validation_idf(blocks)

    if prenom is None or nom is None:
        return {"error": "Failed to extract name information."}

    clean = remove_name_blocks_idf(clean, prenom, nom)

    lieu = extract_lieu_idf(blocks, clean)

    result = {
        "pays": "Maroc",
        "type_de_carte": "Carte Nationale d'Identité",
        "prenom": prenom,
        "Nom": nom,
        "date_de_naissance": dates.get("date_de_naissance"),
        "valable_jusqua": dates.get("valable_jusqua"),
        "lieu_de_naissance": lieu,
        "CIN": (cin[0] if len(cin) != 0 else None)
    }

    return result