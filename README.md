# Contract Intelligence Web App (Detailed Documentation)

**Overview:** This AI-powered web application automates contract review and analytics. It uses OCR, NLP, and ML to convert unstructured contracts (PDFs or scanned images) into actionable insights. It **extracts key clauses, detects fraud indicators, assesses financial terms, and computes a risk/compliance score**. In effect, it helps legal and business teams **speed up reviews, improve compliance, and reduce risk**【5†L178-L186】. The target users are legal practitioners, contract managers, and business analysts who need quick, data-driven contract insights without manual labor.

## 🔥 Features

- **Document Ingestion & OCR:** Users upload contracts (PDF or image). Text is extracted automatically. We use PyMuPDF for PDFs and Tesseract (via pytesseract) for images【5†L231-L238】【19†L19-L27】. Tesseract is a **powerful open-source OCR engine** that is “one of the most popular and powerful” tools for converting scanned text to digital text【19†L19-L27】, making it ideal for digitizing contracts and other documents【37†L449-L457】.  
- **NLP-Based Classification:** The app classifies the contract type (e.g. NDA, SLA) using NLP. We load a spaCy model (`en_core_web_sm`) for general language processing【23†L28-L32】. SpaCy is an industrial-strength NLP library “designed to help you build real products” and excels at large-scale information extraction【23†L28-L32】. We also use Hugging Face’s zero-shot-classification pipeline to label contracts by type on the fly (without custom training)【21†L50-L58】.  
- **Clause Identification & Risk Scoring:** Key clauses (e.g. termination, liability) are identified. The system flags unusual or high-risk provisions. It assigns a **Contract Intelligence Score (CIS)** or risk level by combining clause analysis and historical benchmarks. (In contract analytics, automated tools often “flag deviations” and “assign risk scores” to expedite review【5†L178-L186】.)  
- **Fraud Detection:** The app scans text for fraud indicators. It uses keyword matching and a zero-shot model to spot “fraud” vs. “legitimate” contexts【21†L50-L58】. For example, using Transformers’ pipeline, we do: 
  ```python
  from transformers import pipeline
  fraud_detector = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
  result = fraud_detector(text, candidate_labels=['fraud','legitimate'])
  ``` 
  This leverages a pre-trained model to classify unseen inputs【21†L93-L100】. Suspicious clauses (e.g. hidden fees, excessive penalties) are highlighted.  
- **Financial Data Extraction:** Using regex and NLP, the app pulls out monetary amounts, dates, and payment terms. This supports cash-flow analysis and Net Present Value (NPV) calculations.  
- **Renewal & Deadline Tracking:** Important dates (e.g. renewal deadlines, termination dates) are parsed and summarized so users don’t miss key obligations.  
- **Interactive Visualization:** Results are presented with charts and tables. We use Matplotlib and Seaborn for plotting. Matplotlib is “a Python library for creating static, animated and interactive data visualizations”【31†L28-L31】. Seaborn (built on Matplotlib) provides a high-level interface for attractive statistical graphics【33†L44-L46】. The app displays, for example, bar charts of cash flows, pie charts of risk categories, and line graphs of timeline analyses.  
- **Exportable Reports:** Users can view the analysis on the web UI and download summaries or data tables as needed.

## 🛠️ Tech Stack & Dependencies

- **Python 3.x** – Main programming language.  
- **Streamlit** – Web framework for the UI. Streamlit “turns your data scripts into shareable web apps in minutes”【27†L50-L54】. It allows us to build an interactive front-end with just Python (no HTML/JS).  
- **spaCy** – NLP library (for tokenization, NER, etc.)【23†L28-L32】. We use the `en_core_web_sm` model for English.  
- **Hugging Face Transformers** – Provides the zero-shot classification pipeline【21†L93-L100】. We rely on models like `facebook/bart-large-mnli` for flexible text classification.  
- **PyMuPDF (fitz)** – To read and extract text from PDF files.  
- **pytesseract & Tesseract OCR** – For text extraction from images. (Tesseract is open-source and must be installed separately【37†L401-L404】.)  
- **OpenCV (cv2) & Pillow** – For image preprocessing if needed (e.g. enhancing scanned pages).  
- **Pandas & NumPy** – Data handling (tables, numeric arrays).  
- **Matplotlib & Seaborn** – Plotting and visualization【31†L28-L31】【33†L44-L46】.  
- **Other:** `pickle`, `json` for serialization; `re` for regex; `datetime` for date calculations; `tempfile` for handling uploads.  

## 📥 Installation & Setup

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/your-username/contract-intelligence.git
   cd contract-intelligence
   ```

2. **Install Python Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
   Ensure this file includes packages like `streamlit`, `spacy`, `transformers`, `pymupdf`, `pytesseract`, `opencv-python`, `pandas`, `numpy`, `matplotlib`, `seaborn`, etc.  

3. **Install and Download Models:**  
   - **SpaCy Model:**  
     ```bash
     python -m spacy download en_core_web_sm
     ```  
   - **Hugging Face Models:** The Transformers pipeline will download required model weights (e.g. `facebook/bart-large-mnli`) automatically on first run.  

4. **Install Tesseract OCR:**  
   - **Windows:** Download the installer from [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract). Add the Tesseract installation directory to your system PATH.  
   - **Linux (Ubuntu/Debian):**  
     ```bash
     sudo apt update
     sudo apt install tesseract-ocr
     ```  
     *Example:* On Debian/Ubuntu systems, you can run `sudo apt install tesseract-ocr`【37†L401-L404】 to install Tesseract.  
   - **macOS:** Use Homebrew: `brew install tesseract`.  

5. **Run the App:**  
   ```bash
   streamlit run app.py
   ```  
   This will start a local web server. Open the given URL (typically `http://localhost:8501`) in your browser.

## 🚀 How It Works (Architecture & Data Flow)

The app’s workflow follows a typical **contract analysis pipeline**【9†L176-L184】:

1. **User Input:** User uploads a contract file (PDF or image) via the Streamlit UI.  
2. **Document Ingestion:**  
   - **PDFs:** PyMuPDF extracts text directly from PDF pages.  
   - **Images/Scans:** The image is processed (grayscale conversion, thresholding, etc.) and passed to Tesseract OCR to convert it into text. Tesseract “analyses the shapes of the characters and converts them into digital text”【37†L449-L457】, making even scanned documents searchable.  
3. **Structure Analysis (if any):** We determine section breaks or clause boundaries via pattern matching (e.g. looking for section headers or numbering).  
4. **NLP Processing:** The combined text is fed into the spaCy pipeline: tokenization, POS tagging, and NER to recognize entities (dates, amounts, parties). Key clauses are identified by keyword/regex matching and by running the zero-shot classifier (e.g. “Is this clause about confidentiality or termination?”). This is akin to the “Legal Language Processing” step【9†L176-L184】.  
5. **Machine Learning Analysis:**  
   - **Contract Classification:** Using the zero-shot pipeline, we classify the entire text into categories (e.g. NDA vs. employment contract) without task-specific training【21†L50-L58】.  
   - **Fraud/Risk Classification:** Another zero-shot or rule-based check flags suspicious language. For instance, if the model assigns high confidence to the “fraud” label, we warn the user of a potential issue.  
6. **Data Extraction:** Regex and NLP are used to pull out structured data: monetary values, dates, liabilities, and obligations. For example, all dollar amounts (`\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?`) and dates (e.g. `MM/DD/YYYY`) are extracted.  
7. **Analysis & Scoring:** The system computes metrics such as the **Contract Intelligence Score (CIS)**, risk level, and projected cash flows. It may apply domain logic or simple financial formulas (e.g. present value calculations).  
8. **Visualization & Output:** Results are rendered in the web app. Charts (bar, pie, line) summarize the findings. Examples: 
   - A bar chart of cash flows over time (using Matplotlib/Seaborn【31†L28-L31】【33†L44-L46】).  
   - A summary table of extracted clauses and their classifications.  
   - Highlighted contract excerpts for flagged items (shown in the UI).  

Throughout, the app maintains a simple, responsive interface via Streamlit. The user can iterate by uploading different contracts and immediately seeing updated insights.

## 📸 Usage Example & Sample Output

After starting the app with `streamlit run app.py`, the browser interface will prompt:

- **Upload Panel:** A sidebar or area where you upload a PDF/image file.  
- **Processing Status:** As the file is processed, the app may display progress or messages (e.g. “Extracting text...”, “Analyzing clauses...”).
- **Result Display:** Once done, you’ll see sections like:
  - **Contract Type:** e.g. *“NDA (Non-Disclosure Agreement)”*.  
  - **Fraud Alerts:** e.g. *“Suspicious clause detected: hidden fee clause”*.  
  - **Key Dates:** e.g. *“Termination Date: 12/31/2024”*.  
  - **Financial Impact:** e.g. *“Estimated Liability: \$50,000”*.  
  - **Risk Score:** e.g. *“Risk Level: Medium (55/100)”*.  
  - **Visuals:** Charts such as a cash flow timeline or risk breakdown pie chart.  

*Example Code Snippet:* To illustrate the classification step, the core Python code might look like:
```python
from transformers import pipeline
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
labels = ["NDA", "Employment Contract", "Sales Agreement"]
result = classifier(contract_text, candidate_labels=labels)
print(result['labels'][0], result['scores'][0])
```
This zero-shot approach allows the app to categorize the text into custom labels without pre-training【21†L93-L100】.



## 📂 Project Structure (example)
```
├── app.py                     # Main Streamlit app
├── contract_classifier.py     # NLP & ML functions
├── fraud_detector.py         # Fraud analysis logic
├── cash_flow_extraction.py   # Regex rules for financials
├── requirements.txt          # Python dependencies
├── model/                    # (Optional) serialized models or data
└── README.md                 # This documentation
```

## 🤝 Contributing

Contributions are welcome! If you find issues or have ideas for features, please open an issue or pull request on the GitHub repo. Guidelines:
- Fork the repo and create a feature branch.
- Write clear commit messages and include tests if possible.
- Ensure any added functionality is documented.

## 📜 License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 👤 Author & Contact

**Satya** – *Project Lead / Developer*  
For questions or collaboration, feel free to open an issue on GitHub or contact via the project’s repo.  

---

**Sources:** This document references industry resources to explain technologies used. For example, AI-driven contract analysis tools leverage NLP and OCR to auto-extract clauses and risk scores【5†L178-L186】. Streamlit is an open-source Python framework for quick data app development【27†L50-L54】. Matplotlib and Seaborn enable advanced plotting of results【31†L28-L31】【33†L44-L46】.
