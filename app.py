import streamlit as st
import numpy as np
import tensorflow as tf
from keras.saving import load_model
from PIL import Image

# Load the .keras model
model = load_model("my_model.keras")

# Define preprocessing
def preprocess_image(uploaded_image):
    IMAGE_SIZE = (150, 150)  # Model's expected input size
    img = Image.open(uploaded_image)
    img = img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app

st.title("Image Classification with Keras Model")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(uploaded_image)
    
    # Make a prediction
    prediction = model.predict(img_array)
    class_names = ['Layak', 'Tidak-Layak']
    pred_label = class_names[int(prediction > 0.5)]  # Binary classification threshold
    
    # Display the prediction
    st.write(f"Prediction: {pred_label} menerima Bansos")
    st.write(f"Confidence: {prediction[0][0]:.4f}")
