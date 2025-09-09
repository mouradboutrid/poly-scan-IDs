🪪 ID Document Information Extractor
A powerful Streamlit-based web application that automatically extracts structured information from various ID documents using OCR, deep learning, and face detection models. The app supports national ID cards, driver's licenses, and passports in French and Arabic.

✨ Features
📸 Upload and process scanned images of ID documents

🧠 Advanced OCR using Doctr

🧾 Automatic document type classification (ID front, ID back, license, passport)

🌐 Multilingual OCR filtering (French and Arabic)

👤 Face detection and cropping

📤 Download extracted data as JSON

📥 Download extracted face image

📊 Clean, styled UI with real-time feedback

📂 Supported Document Types
Moroccan National Identity Cards (Front & Back)

Moroccan Driver's Licenses

Moroccan Passports

🛠️ Tech Stack
Technology	Description
Streamlit	For building the interactive web UI
Doctr (OCR)	For text detection and recognition
PyTorch	For the language classifier model
OpenCV	Image processing and face detection
PIL (Pillow)	Image enhancements
Custom Modules	For document-specific parsing and layout
📷 Sample Workflow
Upload an image of a document (.jpg, .jpeg, or .png)

Click the "Extract Information" button

View:

The document with detected text overlays

Detected face (if present)

Parsed personal data displayed in styled card format

Download the:

Extracted JSON data

Cropped face image (if available)

📁 Project Structure (Modules)
Module	Purpose
craft_ocr.py	Text box detection using CRAFT or similar model
id_frt.py	Extraction logic for front of national ID
id_bck.py	Extraction logic for back of national ID
perm_con.py	Extraction logic for driver's licenses
passp.py	Extraction logic for passports
type_classf.py	Document type classification
face_dtc.py	Face detection logic using OpenCV
🧪 Models Used
Model	Purpose	File Path
doctr_model.pth	OCR predictor	/models/doctr_model.pth
language_classifier.pth	Classify OCR language (Arabic vs French)	/models/language_classifier.pth
💡 Tips for Best Results
Use high-resolution images

Ensure good lighting and minimal glare

Place the document on a dark, non-reflective background

Avoid skewed or rotated documents

📦 Installation
You'll need Python 3.8+ installed

bash
# 1. Clone the repository
git clone https://github.com/yourusername/id-document-extractor.git
cd id-document-extractor

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
⚠️ Note: Ensure you place your models in the correct models/ folder as expected in the code.

📤 Example Output
json
{
  "nom": "BENALI",
  "prenom": "Youssef",
  "date_de_naissance": "1992-07-12",
  "CIN": "AB123456",
  "pays": "Maroc",
  "valable_jusqua": "2030-07-12"
}
📜 License
This project is licensed under the MIT License.

🤝 Contributing
Contributions are welcome! Please open issues or submit pull requests to improve the app.

🙋‍♂️ Acknowledgements
Doctr OCR by Mindee

Streamlit

OpenCV

Your custom-trained models and OCR pipeline
