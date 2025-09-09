# POLY-SCAN-IDs

A Streamlit-based web application that automatically extracts structured information from various ID documents using OCR, deep learning, and face detection models. The app supports national ID cards, driver's licenses, and passports in French.

## Features

- Upload and process scanned images of ID documents
- Advanced OCR using Doctr
- Automatic document type classification (ID front, ID back, license, passport)
- OCR filtering (French and Arabic)
- Face detection and cropping
- Download extracted data as JSON
- Download extracted face image
- Clean, styled UI

## Supported Document Types

- Moroccan National Identity Cards (Front & Back)
- Moroccan Driver's Licenses
- Moroccan Passports

## Tech Stack
| Technology | Description |
|------------|-------------|
| Streamlit | For building the interactive web UI |
| Doctr (OCR) | For text detection and recognition |
| PyTorch | For the language classifier model |
| OpenCV | Image processing |
| YOLO | Face detection |
| PIL (Pillow) | Image enhancements |
| Custom Modules | For document-specific parsing and layout |

## Sample Workflow

1. Upload an image of a document (.jpg, .jpeg, or .png)
2. Click the "Extract Information" button
3. View:
   - The document with detected text overlays
   - Detected face (if present)
   - Parsed personal data displayed in styled card format
4. Download the:
   - Extracted JSON data
   - Cropped face image (if available)

## Project Structure (Modules)

| Module | Purpose |
|--------|---------|
| craft_ocr.py | Text box detection using CRAFT or similar model |
| id_frt.py | Extraction logic for front of national ID |
| id_bck.py | Extraction logic for back of national ID |
| perm_con.py | Extraction logic for driver's licenses |
| passp.py | Extraction logic for passports |
| type_classf.py | Document type classification (Model training) |
| face_dtc.py | Face detection logic using YOLO |

## Models Used

| Model | Purpose | File Path |
|-------|---------|-----------|
| doctr_model.pth | OCR predictor | /models/doctr_model.pth |
| language_classifier.pth | Classify OCR language (Arabic vs French) | /models/language_classifier.pth |
| yolov8x.pt | Face detection model | /models/yolov8x.pt (Or auto-Download) |


## Tips for Best Results

- Use high-resolution images
- Ensure good lighting and minimal glare
- Place the document on a dark, non-reflective background
- Avoid skewed or rotated documents (Important)

## Installation

You'll need Python 3.8+ installed

```bash
# 1. Clone the repository
git clone https://github.com/mouradboutrid/poly-scan-IDs.git
cd id-document-extractor

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
