import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image

# Load the model and class indices
model_path = 'my_model.sav'  # Adjust the path if needed
class_indices_path = 'class_indices.sav'

# Load model
model = pickle.load(open(model_path, 'rb'))

# Load class indices
with open(class_indices_path, 'rb') as f:
    class_indices = pickle.load(f)
CATEGORIES = list(class_indices.keys())

st.title("Malware Image Classification")
st.write("Upload an image to check for malware classification.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"])
uploaded_file = uploaded_file.reshape(-1, 64, 64, 3)  # Ensure correct batch dimension
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = image.resize((64, 64))
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape for model input
    
    # Make prediction
    prediction = model.predict(img)
    pred_name = CATEGORIES[np.argmax(prediction)]
    
    st.subheader("Prediction:")
    st.write(f"The uploaded image is classified as: **{pred_name}**")
    
    # Show probabilities
    st.subheader("Prediction Probabilities:")
    for category, prob in zip(CATEGORIES, prediction[0]):
        st.write(f"{category}: {prob:.4f}")
