import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pickle

# Load the model properly
model_path = 'my_model.h5'  # Ensure this is the correct model file
class_indices_path = 'class_indices.sav'

# Load TensorFlow model
model = tf.keras.models.load_model(model_path)

# Load class indices
with open(class_indices_path, 'rb') as f:
    class_indices = pickle.load(f)
CATEGORIES = list(class_indices.keys())

st.title("Malware Image Classification")
st.write("Upload an image to check for malware classification.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Ensure image is RGB (important for model consistency)
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((64, 64))
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    pred_name = CATEGORIES[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(f"The uploaded image is classified as: **{pred_name}**")

    # Show probabilities
    st.subheader("Prediction Probabilities:")
    for category, prob in zip(CATEGORIES, prediction[0]):
        st.write(f"{category}: {prob:.4f}")
