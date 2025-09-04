import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

# Download & load model dari Hugging Face
model_path = hf_hub_download(
    repo_id="zahratalitha/anjingkucing",
    filename="kucinganjing.keras"
)
model = tf.keras.models.load_model(model_path, compile=False)
st.write("Input shape model:", model.input_shape)

# Preprocessing image
def preprocess(img):
    target_size = model.input_shape[1:3]
    img = img.convert("RGB")  # ✅ selalu paksa jadi 3 channel (RGB)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload file
uploaded_file = st.file_uploader("Upload gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess & predict
    input_img = preprocess(image)
    pred = model.predict(input_img)

    # Handle sigmoid (binary) dan softmax (multi-class)
    if pred.shape[1] == 1:  
        prob = float(pred[0][0])
        label = "Kucing 🐱" if prob < 0.5 else "Anjing 🐶"
        confidence = 1 - prob if prob < 0.5 else prob
    else:  
        class_idx = np.argmax(pred[0])
        label = "Kucing 🐱" if class_idx == 0 else "Anjing 🐶"
        confidence = float(np.max(pred[0]))

    st.subheader(f"Prediksi: {label}")
    st.write(f"Confidence: {confidence:.2f}")
