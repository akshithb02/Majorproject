import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the Keras model
model_path = "my_model.h5"  # Ensure it's a .h5 file (not .sav)
model = tf.keras.models.load_model(model_path)

# Define class categories
CATEGORIES = ["Malware", "Benign"]  # Update based on your model's classes

st.title("Malware Image Classification")
st.write("Upload an image to classify whether it contains malware.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    # Convert file to image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((64, 64))  # Resize to match model input
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    # Make prediction
    prediction = model.predict(img)
    pred_index = np.argmax(prediction)
    pred_name = CATEGORIES[pred_index]

    # Display Prediction
    st.subheader("Prediction:")
    st.write(f"The uploaded image is classified as: **{pred_name}**")

    # Show probabilities
    st.subheader("Prediction Probabilities:")
    for category, prob in zip(CATEGORIES, prediction[0]):
        st.write(f"{category}: {prob:.4f}")
