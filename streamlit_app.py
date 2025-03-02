import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pickle

# Load model
model_path = 'my_model.h5'
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices_path = 'class_indices.sav'
with open(class_indices_path, 'rb') as f:
    class_indices = pickle.load(f)
CATEGORIES = list(class_indices.keys())

st.title("Malware Image Classification")
st.write("Upload an image to check for malware classification.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Ensure RGB format
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((64, 64))
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Debugging: Check input shape
    st.write("Input Image Shape:", img.shape)

    # Make prediction
    prediction = model.predict(img)

    # Debugging: Print raw predictions
    st.write("Raw Model Output:", prediction)

    pred_name = CATEGORIES[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(f"The uploaded image is classified as: **{pred_name}**")

    # Show class index mapping
    st.write("Class Indices Mapping:", class_indices)

    # Show probabilities
    st.subheader("Prediction Probabilities:")
    for category, prob in zip(CATEGORIES, prediction[0]):
        st.write(f"{category}: {prob:.4f}")
