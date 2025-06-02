import streamlit as st
from tensorflow import keras
import json
from utils import preprocess_image  # your preprocessing script
import numpy as np

# Load model and labels
model = keras.models.load_model("final_skin_model.h5")
with open("data/label_map.json", "r") as f:
    label_map = json.load(f)

# Correct class mapping (from your training generator)
class_indices = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}

# Reverse it for prediction display
reverse_map = {v: k for k, v in class_indices.items()}


st.title("Skin Disease Prediction")
st.write("Upload a skin lesion image to predict the condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("temp.jpg", caption="Uploaded Image", use_container_width=True)
    
    img_array = preprocess_image("temp.jpg")
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = reverse_map.get(int(predicted_class), "Unknown")


    st.success(f"Predicted Skin Condition: **{predicted_label}**")