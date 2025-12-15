import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from ui import apply_css, display_prediction  # Import UI function


# apply_css(theme="Warm")
apply_css()

# -------------------------------
model = tf.keras.models.load_model("male_female_v2.keras")
class_names = ["Female", "Male"]

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Male vs Female Classifier", layout="wide")
apply_css()  # Apply CSS from ui.py

banner = Image.open("banner1.jpg")
st.image(banner, use_container_width=True)

st.sidebar.title("Upload Images")
uploaded_files = st.sidebar.file_uploader(
    "Choose one or more images...", type=["jpg","jpeg","png"], accept_multiple_files=True
)

# -------------------------------
# Preprocess Function
# -------------------------------
def preprocess_image(image, target_size=(224,224)):
    img_array = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch, img_array

# -------------------------------
# Prediction & Display
# -------------------------------
if uploaded_files:
    st.subheader("Predictions")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        img_input, img_array = preprocess_image(image)

        prediction = model.predict(img_input)[0]


        if len(prediction) == 1:  # sigmoid
            male_prob = float(prediction[0])
            female_prob = 1.0 - male_prob

            probs = [female_prob, male_prob]
            class_index = int(male_prob > 0.5)
            confidence = probs[class_index]

        else:  # softmax
            class_index = int(np.argmax(prediction))
            confidence = float(prediction[class_index])
            probs = [float(p) for p in prediction]

        predicted_label = class_names[class_index]

        display_prediction(img_array, predicted_label, confidence, probs, class_names)

