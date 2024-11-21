import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 255

# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Streamlit app
st.title("Potato Disease Classification")
st.write("Upload an image of a potato leaf to classify it as Early Blight, Late Blight, or Healthy.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and display the image
    image_path = os.path.join("temp", uploaded_file.name)
    img = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict and display the results
    st.write("Classifying...")
    predicted_class, confidence = predict(img)
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence}%")
