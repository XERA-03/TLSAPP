# 🚀 Contract Intelligence Web App

**Overview:** This AI-powered web application automates contract review and analytics. It uses OCR, NLP, and ML to convert unstructured contracts (PDFs or scanned images) into actionable insights. It **extracts key clauses, detects fraud indicators, assesses financial terms, and computes a risk/compliance score**. In effect, it helps legal and business teams **speed up reviews, improve compliance, and reduce risk**. The target users are legal practitioners, contract managers, and business analysts who need quick, data-driven contract insights without manual labor.

Under the hood, the app uses SpaCy (an open-source NLP library) and pre-trained transformers. For example, a **zero-shot classifier** labels a contract with custom categories (NDA, SLA, etc.) without needing special training. It also runs a fraud-detection check by scanning for suspicious clauses. The result is an interactive report: charts of cash flows, a list of extracted clauses with metadata, and a summarized **Contract Intelligence Score (CIS)** – all presented through a user-friendly interface.

## 🔥 Features

- **Document Ingestion & OCR:** Users upload contracts (PDF or image). Text is extracted automatically using PyMuPDF for PDFs and Tesseract OCR for images. Tesseract is a **powerful open-source OCR engine** that is “one of the most popular and powerful” tools for converting scanned text to digital text, making it ideal for digitizing contracts.  
- **NLP-Based Classification:** The app classifies the contract type (e.g. NDA, SLA) using NLP. We use spaCy (with `en_core_web_sm`) for base language processing and Hugging Face’s zero-shot-classification pipeline to label documents on the fly.  
- **Clause Identification & Risk Scoring:** Key clauses (e.g. termination, liability) are automatically identified and flagged. The system highlights unusual or high-risk terms, and computes a **Contract Intelligence Score** by combining clause analysis with historical benchmarks. Modern contract analytics similarly “flags deviations” and “assigns risk scores” to expedite reviews.  
- **Fraud Detection:** The app scans for fraud indicators using keyword checks and AI. For example, a zero-shot classifier distinguishes “fraud” vs. “legitimate” contexts. Suspicious clauses (hidden fees, onerous penalties) are highlighted for review.  
- **Financial Data Extraction:** Regex and NLP extract monetary amounts, dates, and payment terms. This data enables cash-flow projections and Net Present Value calculations.  
- **Renewal & Deadline Tracking:** Important dates (renewals, expirations) are parsed and summarized to ensure no obligations are missed.  
- **Interactive Visualization:** The results are presented with charts and tables. We use Matplotlib and Seaborn for plotting. Matplotlib is “a Python library for creating static, animated and interactive data visualizations”, and Seaborn provides a high-level interface for informative statistical graphics. For example, the app shows bar charts of financial obligations and pie charts of risk factors.  
- **Exportable Reports:** Users can save or download summaries of the analysis directly from the app.

## 🛠️ Tech Stack & Dependencies

- **Python 3** – Core language.  
- **Streamlit** – Web framework for the UI. Streamlit “turns your data scripts into shareable web apps in minutes”. It lets us build an interactive dashboard with Python only.  
- **spaCy** – NLP library (tokenization, POS tagging, NER, etc.), using the `en_core_web_sm` model.  
- **Hugging Face Transformers** – Provides the zero-shot-classification pipeline (we use models like `facebook/bart-large-mnli`).  
- **PyMuPDF (fitz)** – PDF reading and text extraction.  
- **pytesseract & Tesseract OCR** – For text extraction from images (scans). Tesseract is open-source and must be installed separately.  
- **OpenCV (cv2)** & **Pillow** – Image preprocessing (e.g. resizing, thresholding).  
- **Pandas & NumPy** – Data handling (tables, numeric arrays).  
- **Matplotlib & Seaborn** – Plotting libraries for charts.  
- **Others:** `pickle`, `json` (serialization), `re` (regex), `datetime` (dates), `tempfile` (temporary files).

## 📥 Installation & Setup

1. **Clone the repository:**  
 

2. **Install Python dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  
   Ensure `requirements.txt` includes packages like `streamlit`, `spacy`, `transformers`, `PyMuPDF`, `pytesseract`, `opencv-python`, `pandas`, `numpy`, `matplotlib`, `seaborn`, etc.

3. **Download models:**  
   - SpaCy model: `python -m spacy download en_core_web_sm`.  
   - Transformers models will auto-download on first run (e.g. `facebook/bart-large-mnli` for zero-shot).

4. **Install Tesseract OCR:**  
   - **Windows:** Install from [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract) and add to PATH.  
   - **Linux (Ubuntu/Debian):**  
     ```bash
     sudo apt update
     sudo apt install tesseract-ocr
     ```  
     (Example: on Debian/Ubuntu you can run `sudo apt install tesseract-ocr`.)  
   - **macOS:** `brew install tesseract`.

5. **Run the app:**  
   ```bash
   streamlit run app.py
   ```  
   Then open `http://localhost:8501` in a browser to use the app.

## 🚀 How It Works (Architecture & Data Flow)

The app follows a standard **contract analysis pipeline**:

1. **Upload Contract:** User uploads a PDF or image.  
2. **Text Extraction:**  
   - **PDFs:** PyMuPDF reads page text.  
   - **Images:** Pre-process (grayscale, binarize) and use Tesseract OCR to get text. Tesseract “analyses the shapes of the characters and converts them into digital text”.  
3. **NLP Processing:** Text is tokenized and analyzed with spaCy. Named entities (dates, amounts, parties) are recognized, and clauses are segmented (e.g., by headers or regex).  
4. **AI Classification:**  
   - A zero-shot classifier tags the document type (e.g. NDA vs. loan agreement).  
   - A second classifier scans for fraud or high-risk language.  
5. **Data Extraction:** Regex and NLP extract structured data: financial amounts, deadlines, obligations.  
6. **Analysis & Output:** The system calculates metrics (risk scores, cash flows) and generates plots (bar charts, timelines). All results—contract category, detected issues, scores, charts—are displayed in the UI.

This pipeline mirrors best practices in contract intelligence, combining OCR, NLP, and machine learning to produce actionable insights.

## 📊 Example Output

After running the app and uploading a contract, the user sees results such as:  
- **Contract Type:** e.g. *“NDA (Non-Disclosure Agreement)”*.  
- **Alerts:** e.g. *“Detected one-sided liability clause”*.  
- **Key Dates:** e.g. *“Termination Date: 12/31/2024”*.  
- **Financial Impact:** e.g. *“Estimated liability: \$50,000”*.  
- **Risk Score:** e.g. *“CIS Score: 78/100 (Medium Risk)”*.  
- **Visuals:** Charts of cash flows, clauses per category, etc.

*Sample snippet:* A Python example of zero-shot classification for contract type:
```python
from transformers import pipeline
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
labels = ["NDA", "Employment", "Sales Agreement"]
result = classifier(contract_text, candidate_labels=labels)
print("Predicted category:", result['labels'][0])
```
This matches text against labels without special training.

## 📂 Project Structure (example)
```
├── app.py                  
├── contract_analysis.py    
├── fraud_detector.py       
├── cash_flow_extraction.py 
├── requirements.txt        
└── README.md               
```

## 🤝 Contributing

Contributions are welcome. To contribute:  
- Fork the repository and create a new branch.  
- Submit issues for bugs or feature ideas.  
- Open pull requests with descriptive commits.  

Please follow standard git and GitHub workflows for a smooth process.

## 📜 License

This project is released under the **MIT License**. See the `LICENSE` file for details.

## 👤 Author & Contact

**Satya** – Project Lead/Developer  
For inquiries or feedback, open an issue or contact via the project GitHub.

---

**Sources:** This README pulls from contract analytics and NLP resources. For example, contract intelligence often uses NLP/ML to automate clause extraction and risk assessment. Streamlit’s docs note how easily it builds data apps. Matplotlib/Seaborn documentation highlights their plotting capabilities. All the above info is used to describe and justify the features of this app.
