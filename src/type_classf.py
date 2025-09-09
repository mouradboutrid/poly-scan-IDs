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

def normalize(text):
    text = unicodedata.normalize("NFD", text.lower().strip())
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

def classify_card_type(blocks):

    all_texts = [normalize(block.get('text', '')) for block in blocks]

    date_pattern = re.compile(r'\b\d{2}[\s./-]\d{2}[\s./-]\d{4}\b')
    date_count = sum(len(date_pattern.findall(text)) for text in all_texts)

    has_id_card = False
    has_permis = False
    has_passport = False

    for text in all_texts:
        if text == normalize("CARTE NATIONALE D'IDENTITE"):
            has_id_card = True
        elif "carte nationale" in text:
            has_id_card = True
        elif "valable jusqu" in text:
            has_id_card = True
        elif text == normalize("PERMIS DE CONDUIRE"):
            has_permis = True
        elif "conduire" in text:
            has_permis = True
        elif "passeport" in text:
            has_passport = True
        elif "kingdom of morocco" in text or "kingdom" in text:
            has_passport = True

    if has_passport:
        return "passport"
    if has_permis:
        return "permis_de_conduire"
    if date_count == 0:
        return "id_card_back"
    if date_count >= 3:
        return "passport"
    if has_id_card:
        return "id_card_front"
    if date_count <= 2:
        return "id_card_front"
    return "id_card_back"