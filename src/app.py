
import streamlit as st
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
import tensorflow as tf
import re
import torch
from datetime import datetime
import json
import difflib
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import torch.nn as nn
from torchvision import transforms
from craft_ocr import *
from id_frt import *
from id_bck import *
from perm_con import *
from passp import *
from type_classf import *
from face_dtc import *
import tempfile
import io

# === Initialize models (keep your existing loading code) ===
model_path_doctr = '/content/drive/MyDrive/poly-scan-ID/models/doctr_model.pth'
config_path_doctr = '/content/drive/MyDrive/poly-scan-ID/models/doctr_model_config.json'

predictor = load_model(model_path_doctr, config_path_doctr)

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model_path_classifier = '/content/drive/MyDrive/poly-scan-ID/models/language_classifier.pth'
num_classes = 2  # Arabic and French

model = LanguageClassifier(num_classes)
model.load_state_dict(torch.load(model_path_classifier, map_location=torch.device('cpu')))
model.eval()

# === Custom CSS for ID card styling ===
st.markdown("""
<style>
    .id-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        border: 2px solid #2c3e50;
        margin-left: 0;
        max-width: 100%;
    }
    .id-header {
        text-align: center;
        color: #2c3e50;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 25px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .id-section {
        background: white;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-left: 5px solid #3498db;
    }
    .id-field {
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 1px dashed #e0e0e0;
        font-size: 18px;
    }
    .id-label {
        font-weight: bold;
        color: #2c3e50;
        min-width: 200px;
        font-size: 18px;
        text-align: left;
    }
    .id-value {
        color: #2c3e50;
        text-align: left;
        flex-grow: 1;
        padding-left: 20px;
        font-size: 18px;
    }
    .empty-value {
        color: #95a5a6;
        font-style: italic;
    }
    .face-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .document-image {
        border-radius: 10px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        max-width: 80%;
    }
    .small-image {
        max-width: 300px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton button {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px 30px;
        font-weight: bold;
        width: 100%;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .download-btn {
        background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px 30px;
        font-weight: bold;
        width: 100%;
        margin-top: 20px;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .big-text {
        font-size: 20px !important;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #3498db;
        text-align: left;
    }
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
        padding: 15px;
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .center-content {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        flex-direction: column;
        margin-left: 0;
        padding-left: 0;
    }
    .download-section {
        display: flex;
        justify-content: flex-start;
        gap: 20px;
        margin-top: 20px;
        flex-wrap: wrap;
    }
    .download-button {
        background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    .download-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
        text-decoration: none;
    }
    .images-row {
        display: flex;
        justify-content: flex-start;
        gap: 30px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    .image-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        max-width: 300px;
    }
    .image-caption {
        margin-top: 10px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
    }
    .main-content {
        margin-left: 0;
        padding-left: 0;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .extraction-results {
        margin-top: 30px;
    }
    .uploaded-image-container {
        margin-bottom: 30px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# === Streamlit app ===
st.markdown('<div class="main-header">🪪 ID Document Information Extractor</div>', unsafe_allow_html=True)

# Sidebar for file upload and download options
with st.sidebar:
    st.markdown("## 📁 Upload Document")
    uploaded_file = st.file_uploader("Choose an ID document image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Display download options if data is available
    if 'extracted_data' in st.session_state:
        st.markdown("## 💾 Download Results")
        
        # Download JSON button
        json_data = json.dumps(st.session_state.extracted_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name=f"{st.session_state.doc_type}_extracted_data.json",
            mime="application/json",
            use_container_width=True,
            key="download_json"
        )
        
        # Download face image if available
        if 'face_image' in st.session_state and st.session_state.face_image is not None:
            # Convert the face image to bytes
            face_img = st.session_state.face_image
            is_success, buffer = cv2.imencode(".jpg", face_img)
            if is_success:
                io_buf = io.BytesIO(buffer)
                st.download_button(
                    label="Download Face Image",
                    data=io_buf,
                    file_name="extracted_face.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                    key="download_face"
                )
    
    st.markdown("---")
    st.markdown("## Instructions")
    st.info("""
    1. Upload an ID document image
    2. Click 'Extract Information'
    3. View the extracted data
    4. Download as JSON or face image if needed
    """)
    
    st.markdown("---")
    st.markdown("## Supported Documents")
    st.markdown("""
    - National ID Cards (Front & Back)
    - Driver's Licenses
    - Passports
    """)
    
    st.markdown("---")
    st.markdown("## Tips")
    st.markdown("""
    For best results:
    - Ensure good lighting
    - Place document on a dark background
    - Make sure all text is clearly visible
    """)

# Main content area
if uploaded_file:
    # Display the uploaded image only once at the top
    st.markdown('<div class="uploaded-image-container">', unsafe_allow_html=True)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Document", width=400, output_format="JPEG")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process the image when the user clicks the button
    if st.button("Extract Information", key="extract_btn", use_container_width=True):
        with st.spinner("Processing document... This may take a few moments"):
            # Save uploaded file temporarily to pass path to functions that expect a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name

            # Detect text boxes and get image from detect_text (which returns an image, boxes, polys)
            try:
                image_cv, boxes, polys = detect_text(predictor, temp_file_path)
            except Exception as e:
                st.error(f"Text detection error: {e}")
                image_cv = None
                boxes = None

            if image_cv is not None:
                try:
                    ocr_results = run_ocr(image_cv, predictor)
                except Exception as e:
                    st.error(f"OCR error: {e}")
                    ocr_results = None

                if ocr_results is not None:
                    try:
                        filtered_texts = filter_ocr_with_classification(
                            ocr_results,
                            image_cv,
                            model,
                            ['Arabic', 'French'],
                            test_transform
                        )
                    except Exception as e:
                        st.error(f"Filtering OCR error: {e}")
                        filtered_texts = None
                else:
                    filtered_texts = None
            else:
                ocr_results = None
                filtered_texts = None

            if filtered_texts is None:
                st.error("No information could be extracted from the document.")
            else:
                # Classify document type
                doc_type = classify_card_type(filtered_texts)
                
                # Extract info depending on document type
                if doc_type == "id_card_front":
                    extracted = extract_id_info(filtered_texts)
                elif doc_type == "id_card_back":
                    extracted = extract_idback_info(filtered_texts)
                elif doc_type == "permis_de_conduire":
                    extracted = extract_permis_info(filtered_texts, ocr_results)
                elif doc_type == "passport":
                    extracted = extract_passport_info(filtered_texts)
                else:
                    extracted = {"error": "Unknown document type"}

                # Face detection (using temp file path)
                try:
                    face_box, face_crop = detect_face(temp_file_path)
                    if face_crop is not None:
                        st.session_state.face_image = face_crop
                    else:
                        st.session_state.face_image = None
                except Exception as e:
                    st.warning(f"Face detection warning: {e}")
                    face_box, face_crop = None, None
                    st.session_state.face_image = None

                # Store extracted data in session state for download
                st.session_state.extracted_data = extracted
                st.session_state.doc_type = doc_type
                
                # Display extraction results
                st.markdown('<div class="extraction-results">', unsafe_allow_html=True)
                
                # Display images side by side only if face is detected
                if face_crop is not None:
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        # Extracted face
                        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        st.image(face_crop_rgb, caption="Extracted Face", width=300)
                    
                    with img_col2:
                        # Text detection visualization
                        st.image(image_cv, caption="Text Detection", width=300)
                
                # Display the ID card interface
                st.markdown('<div class="id-card">', unsafe_allow_html=True)
                
                # Document type header
                doc_type_display = extracted.get("type_de_carte", doc_type.replace("_", " ").title())
                st.markdown(f'<div class="id-header">{doc_type_display}</div>', unsafe_allow_html=True)
                
                # Personal information section
                st.markdown('<div class="id-section">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
                
                # Define fields to display based on document type
                if doc_type == "id_card_front":
                    fields = [
                        ("Pays", extracted.get("pays", "")),
                        ("Prénom", extracted.get("prenom", "")),
                        ("Nom", extracted.get("Nom", "")),
                        ("Date de Naissance", extracted.get("date_de_naissance", "")),
                        ("Lieu de Naissance", extracted.get("lieu_de_naissance", "")),
                        ("CIN", extracted.get("CIN", "")),
                        ("Valable Jusqu'à", extracted.get("valable_jusqua", "")),
                    ]
                elif doc_type == "id_card_back":
                    fields = [
                        ("Pays", extracted.get("pays", "")),
                        ("CIN", extracted.get("CIN", "")),
                        ("Code", extracted.get("Code", "")),
                        ("Numéro Civil", extracted.get("Num_Civil", "")),
                        ("Sexe", extracted.get("Sexe", "")),
                        ("Père", extracted.get("Pere", "")),
                        ("Mère", extracted.get("Mere", "")),
                        ("Adresse", extracted.get("Adresse", ""))
                    ]
                elif doc_type == "permis_de_conduire":
                    fields = [
                        ("Pays", extracted.get("pays", "")),
                        ("Prénom", extracted.get("prenom", "")),
                        ("Nom", extracted.get("nom", "")),
                        ("Lieu de Naissance", extracted.get("lieu_naissance", "")),
                        ("Lieu de Délivrance", extracted.get("lieu_delivrance", "")),
                        ("Numéro de Permis", extracted.get("numero_permis", "")),
                        ("Date de Naissance", extracted.get("date_de_naissance", "")),
                        ("Délivré Le", extracted.get("delivre_le", "")),
                        ("Catégories", ", ".join(extracted.get("categories", [])) if extracted.get("categories") else ""),
                        ("CIN", extracted.get("CIN", "")),
                    ]
                elif doc_type == "passport":
                    fields = [
                        ("Pays", extracted.get("pays", "")),
                        ("Nationalité", extracted.get("Naionalite", "")),
                        ("Numéro de Passeport", extracted.get("numero_passport", "")),
                        ("CIN", extracted.get("cin", "")),
                        ("Date de Naissance", extracted.get("date_de_naissance", "")),
                        ("Date de Délivrance", extracted.get("date_de_delivrance", "")),
                        ("Date d'Expiration", extracted.get("date_dexpiration", "")),
                        ("Autorité", extracted.get("authorite", "")),
                        ("Sexe", extracted.get("sexe", "")),
                        ("Type de Passeport", extracted.get("passport_type", "")),
                        ("Nom", extracted.get("nom", "")),
                        ("Prénom", extracted.get("prenom", "")),
                        ("Lieu de Naissance", extracted.get("place_of_birth", "")),
                        ("Adresse", extracted.get("address", "")),
                    ]
                else:
                    fields = [("Error", extracted.get("error", "Unknown document type"))]
                
                # Display all fields
                for label, value in fields:
                    display_value = value if value else "Not detected"
                    value_class = "id-value" if value else "id-value empty-value"
                    
                    st.markdown(f'''
                    <div class="id-field">
                        <div class="id-label">{label}:</div>
                        <div class="{value_class}">{display_value}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close id-section
                st.markdown('</div>', unsafe_allow_html=True)  # Close id-card
                
                # Download buttons
                st.markdown('<div class="download-section">', unsafe_allow_html=True)
                
                # Download JSON button
                json_data = json.dumps(st.session_state.extracted_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"{st.session_state.doc_type}_extracted_data.json",
                    mime="application/json",
                    key="download_json_main"
                )
                
                # Download face image if available
                if 'face_image' in st.session_state and st.session_state.face_image is not None:
                    face_img = st.session_state.face_image
                    is_success, buffer = cv2.imencode(".jpg", face_img)
                    if is_success:
                        io_buf = io.BytesIO(buffer)
                        st.download_button(
                            label="📸 Download Face",
                            data=io_buf,
                            file_name="extracted_face.jpg",
                            mime="image/jpeg",
                            key="download_face_main"
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close download-section
                
                # Show raw JSON data in expander
                with st.expander("View Raw Extracted Data"):
                    st.json(extracted)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close extraction-results

# Display instructions if no file uploaded
else:
    st.info(" Please upload an ID document image using the sidebar to begin extraction.")
    st.markdown("""
    <div style="background: #f8f9fa; padding: 30px; border-radius: 15px; margin-top: 30px; font-size: 18px;">
        <h3 style="color: #2c3e50;">About This Tool</h3>
        <p>This application extracts information from various ID documents using advanced OCR and machine learning techniques.</p>
        <p>Simply upload your document using the sidebar and click the extract button to see the results.</p>
    </div>
    """, unsafe_allow_html=True)
