import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Download model dari HuggingFace Hub
model_path = hf_hub_download(
    repo_id="zahratalitha/anjingkucing",  # ganti dengan repo_id kamu
    filename="kucinganjing.keras"         # ganti dengan nama file model kamu
)

# Load model
model = tf.keras.models.load_model(model_path, compile=False)

st.title("Klasifikasi Anjing vs Kucing 🐶🐱")

st.write("Upload gambar untuk mengetahui apakah itu kucing atau anjing.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

def preprocess(img):
    img = img.resize((180, 180))              # resize sesuai input model
    img = img.convert("RGB")                  # pastikan 3 channel (RGB)
    img = np.array(img) / 255.0               # normalisasi
    img = np.expand_dims(img, axis=0)         # tambah batch dimension
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # Preprocess gambar
    processed_img = preprocess(image)

    # Prediksi
    preds = model.predict(processed_img)
    label = np.argmax(preds, axis=1)[0]

    if label == 0:
        st.success("🐱 Ini adalah **Kucing**")
    else:
        st.success("🐶 Ini adalah **Anjing**")
