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
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the environment**
   - **Windows (PowerShell):**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download model and scaler files**
   - Download the `models/` folder from this link.  
   - Place the folder at the root of the project so the structure looks like:

   ```
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
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```
