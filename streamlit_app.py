import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pickle

# ------------------- Load Model & Class Indices -------------------
MODEL_PATH = 'my_model.h5'  # Ensure this is a TensorFlow/Keras saved model
CLASS_INDICES_PATH = 'class_indices.sav'

# Load trained model
@st.cache_resource  # Caches the model for faster reloads
def load_malware_model():
    return load_model(MODEL_PATH)

model = load_malware_model()

# Load class indices
with open(CLASS_INDICES_PATH, 'rb') as f:
    class_indices = pickle.load(f)

CATEGORIES = list(class_indices.keys())

# ------------------- Streamlit UI -------------------
st.title("🛡️ Malware Image Classification")
st.write("Upload an image to classify it as a specific type of malware.")

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image file", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
    
    # ----------- Image Preprocessing -----------
    img = image.resize((64, 64))  # Resize to match model input size
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # ----------- Prediction -----------
    prediction = model.predict(img)
    
    # ----------- Debugging: Show Raw Predictions -----------
    st.subheader("Raw Model Predictions:")
    for category, prob in zip(CATEGORIES, prediction[0]):
        st.write(f"{category}: {prob:.4f}")

    # ----------- Display Final Result -----------
    pred_index = np.argmax(prediction)
    pred_name = CATEGORIES[pred_index]

    st.subheader("🔍 Prediction:")
    st.write(f"**🛑 Malware Type:** `{pred_name}`")

    # Optional: Show Confidence Score
    confidence = prediction[0][pred_index] * 100
    st.write(f"🔢 Confidence: `{confidence:.2f}%`")

