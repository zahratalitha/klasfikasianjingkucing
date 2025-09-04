import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

model_path = hf_hub_download(
    repo_id="zahratalitha/anjingkucing",
    filename="kucinganjing.keras"
)
model = tf.keras.models.load_model(model_path, compile=False)
st.write("Input shape model:", model.input_shape)

def preprocess(img):
    target_size = model.input_shape[1:3]
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

uploaded_file = st.file_uploader("Upload gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    input_img = preprocess(image)
    pred = model.predict(input_img)

if pred.shape[1] == 1:  
    label = "Kucing 🐱" if pred[0][0] < 0.5 else "Anjing 🐶"
else:  
    label = "Kucing 🐱" if np.argmax(pred[0]) == 0 else "Anjing 🐶"

st.subheader(f"Prediksi: {label}")
