# Multimodal-Emotion-Classification
Emotion classification through the integration of multimodal data (text, audio, and images).

# Multimodal Emotion Classification

Emotion analysis through the integration of **text**, **audio**, and **image** data using deep learning models.

---

## 📂 Repository Contents

- `app.py` – Main Streamlit application for emotion detection.  
- `requirements.txt` – Required dependencies.  
- `.gitignore` – Prevents large files such as models/scalers from being uploaded to GitHub.  
- `models/` (excluded from repo) – Contains the trained model and scaler `.pkl` files.  
- `data/` (excluded from repo) – Contains lexicon files such as `NRC-VAD-Lexicon.txt`.  

---

## 🚀 Run the Application Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/FisanSyafa/Multimodal-Emotion-Classification.git
   cd Multimodal-Emotion-Classification
Create a virtual environment (optional but recommended)

2. bash
3. 
\python -m venv venv

Activate the environment:


Windows (PowerShell):


3. bash

venv\Scripts\activate

Linux / macOS:


5. bash

source venv/bin/activate

Install dependencies


7. bash

pip install -r requirements.txt

Download model and scaler files

– Download the models/ folder from this link.
– Place the folder at the root of the project so the structure looks like:

9. Structure
   
├── app.py

├── models/

│   ├── best_model_fold_5_savedmodel/

│   ├── scaler_vad.pkl

│   ├── scaler_bert.pkl

│   ├── scaler_audio.pkl

│   └── scaler_image.pkl

├── data/

│   └── NRC-VAD-Lexicon.txt

├── requirements.txt

└── README.md

Run the application


8. bash

streamlit run app.py
