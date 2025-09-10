import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import VGG16  # Perubahan di sini
import torch
import joblib
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import tempfile
import os

# === 1. SETUP AWAL STREAMLIT ===
st.set_page_config(page_title="Multimodal Emotion Detection", layout="wide")
st.title("🎭 Multimodal Emotion Detection System")
st.markdown("""
Analisis emosi dari teks, suara, dan gambar secara bersamaan menggunakan model deep learning.
""")

# === 2. LOAD MODEL & SCALER ===
@st.cache(allow_output_mutation=True)
def load_models():
    """Muat semua model dan scaler dengan caching untuk performa"""
    # Inisialisasi device untuk torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load scalers
    scalers = {
        'vad': joblib.load("models/scaler_vad.pkl"),
        'bert': joblib.load("models/scaler_bert.pkl"),
        'audio': joblib.load("models/scaler_audio.pkl"),
        'image': joblib.load("models/scaler_image.pkl")
    }
    
    # Load BERT model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    
    # Load emotion prediction model
    model = tf.saved_model.load("models/best_model_fold_5_savedmodel")
    
    # Load NRC-VAD Lexicon
    nrc_vad = pd.read_csv("data/NRC-VAD-Lexicon.txt", sep="\t", 
                         names=['Word', 'Valence', 'Arousal', 'Dominance'], header=None)
    
    return {
        'scalers': scalers,
        'tokenizer': tokenizer,
        'bert_model': bert_model,
        'emotion_model': model,
        'nrc_vad': nrc_vad,
        'device': device
    }

# === 3. FUNGSI EKSTRAKSI FITUR ===
def extract_vad_features(text, nrc_vad):
    words = text.lower().split()
    valence, arousal, dominance = [], [], []
    for word in words:
        if word in nrc_vad['Word'].values:
            entry = nrc_vad[nrc_vad['Word'] == word].iloc[0]
            valence.append(entry['Valence'])
            arousal.append(entry['Arousal'])
            dominance.append(entry['Dominance'])
    return np.array([
        np.mean(valence) if valence else 0,
        np.mean(arousal) if arousal else 0,
        np.mean(dominance) if dominance else 0
    ])

@st.cache(allow_output_mutation=True)
def get_vgg16():
    return VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_bert_features(text, tokenizer, bert_model):
    """Ekstraksi fitur teks menggunakan BERT"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_audio_features(file_path, max_pad_len=150):
    """Ekstraksi fitur audio menggunakan MFCC"""
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    def pad_features(feature, max_len=max_pad_len):
        if feature.shape[1] < max_len:
            return np.pad(feature, ((0, 0), (0, max_len - feature.shape[1])), mode='constant')
        else:
            return feature[:, :max_len]

    mfcc = pad_features(mfcc)
    delta_mfcc = pad_features(delta_mfcc)
    delta2_mfcc = pad_features(delta2_mfcc)

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1)
    ])

def load_and_preprocess_image(img_path):
    """Preprocessing gambar untuk model VGG16"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224]) / 255.0
    return img

# === 4. ANTARMUKA PENGGUNA ===
def main():
    # Sidebar untuk upload file
    with st.sidebar:
        st.header("Upload Files")
        text_input = st.text_area("Masukkan teks:", "I feel very happy today!")
        audio_file = st.file_uploader("Upload file audio (WAV/MP3):", type=['wav', 'mp3'])
        image_file = st.file_uploader("Upload gambar:", type=['jpg', 'jpeg', 'png'])
        
        st.markdown("---")
        st.info("Pastikan semua input telah diisi sebelum melakukan prediksi")

    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Preview")
        if text_input:
            st.text_area("Teks yang akan diproses:", value=text_input, height=150)
        
        if audio_file:
            st.audio(audio_file)
            
        if image_file:
            st.image(image_file, caption="Gambar yang diupload", use_column_width=True)
    
    with col2:
        st.subheader("Emotion Prediction")
        if st.button("Predict Emotion"):
            if not all([text_input, audio_file, image_file]):
                st.error("Harap lengkapi semua input!")
            else:
                with st.spinner("Memproses data..."):
                    try:
                        # Simpan file upload ke temporary file
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_audio:
                            tmp_audio.write(audio_file.read())
                            audio_path = tmp_audio.name
                            
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_image:
                            tmp_image.write(image_file.read())
                            image_path = tmp_image.name
                        
                        # Load models
                        models = load_models()
                        
                        # Lakukan prediksi
                        pred = predict_emotion(
                            text_input, 
                            audio_path, 
                            image_path,
                            models
                        )
                        
                        # Tampilkan hasil
                        st.success(f"Predicted Emotion: **{pred.upper()}**")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                    finally:
                        # Cleanup temporary files
                        if 'audio_path' in locals() and os.path.exists(audio_path):
                            os.unlink(audio_path)
                        if 'image_path' in locals() and os.path.exists(image_path):
                            os.unlink(image_path)

# === 5. FUNGSI UTAMA PREDIKSI ===
def predict_emotion(text, audio_path, image_path, models):
    # Ekstraksi fitur teks
    X_text_vad = extract_vad_features(text, models['nrc_vad'])
    X_text_bert = extract_bert_features(text, models['tokenizer'], models['bert_model'])
    
    # Normalisasi
    X_text_vad = models['scalers']['vad'].transform(X_text_vad.reshape(1, -1))
    X_text_bert = models['scalers']['bert'].transform(X_text_bert.reshape(1, -1))
    X_text = np.hstack([X_text_vad, X_text_bert])[:, :100]
    
    # Ekstraksi fitur audio
    X_audio = extract_audio_features(audio_path).reshape(1, -1)
    X_audio = models['scalers']['audio'].transform(X_audio)
    
    # Ekstraksi fitur gambar
    img = load_and_preprocess_image(image_path)
    vgg16 = get_vgg16()
    X_image = vgg16(np.expand_dims(img, axis=0)).numpy().reshape(1, -1)
    X_image = models['scalers']['image'].transform(X_image)[:, :100]
    
    # Prediksi
    inputs = {
        'text_input': tf.convert_to_tensor(X_text, dtype=tf.float32),
        'audio_input': tf.convert_to_tensor(X_audio, dtype=tf.float32),
        'image_input': tf.convert_to_tensor(X_image, dtype=tf.float32)
    }
    
    outputs = models['emotion_model'].signatures['serving_default'](**inputs)
    prediction = outputs['output_0'].numpy()
    
    # Decode label
    label_encoder = LabelEncoder()
    label_encoder.fit(['neutral', 'happy', 'sad', 'surprised', 'angry', 'fear'])
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]

if __name__ == "__main__":
    main()