import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# -------------------------
# Konfigurasi Halaman
# -------------------------
st.set_page_config(page_title="Klasifikasi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

# -------------------------
# Load Model dari HuggingFace Hub
# -------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zahratalitha/anjingkucing",  # ganti dengan repo kamu
        filename="kucinganjing.keras"         # nama file model
    )
    try:
        model = keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={  # kalau ada preprocessing layer
                "Rescaling": keras.layers.Rescaling,
                "Normalization": keras.layers.Normalization,
            }
        )
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()
    return model

model = load_model()
st.success("✅ Model berhasil dimuat")
st.write("Input shape model:", model.input_shape)

# -------------------------
# Fungsi Preprocessing Gambar
# -------------------------
def preprocess(img: Image.Image):
    # Paksa jadi RGB
    img = img.convert("RGB")

    # Resize sesuai input model
    target_size = tuple(model.input_shape[1:3])
    img = img.resize(target_size)

    # Convert ke numpy
    img_array = np.asarray(img, dtype=np.float32) / 255.0

    # Pastikan channel terakhir 3
    if img_array.ndim == 2:  # grayscale nakal
        img_array = np.stack([img_array] * 3, axis=-1)

    return np.expand_dims(img_array, axis=0)

# -------------------------
# Upload Gambar
# -------------------------
uploaded_file = st.file_uploader("📤 Upload gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Gambar yang diupload", use_column_width=True)

    try:
        # Preprocessing
        input_img = preprocess(image)

        # Prediksi
        pred = model.predict(input_img)

        # Binary classification (sigmoid) atau multi-class (softmax)
        if pred.shape[1] == 1:  # sigmoid
            prob = float(pred[0][0])
            probs = [1 - prob, prob]  # [kucing, anjing]
        else:  # softmax
            probs = pred[0].tolist()

        labels = ["Kucing 🐱", "Anjing 🐶"]
        class_idx = int(np.argmax(probs))
        label = labels[class_idx]
        confidence = float(np.max(probs))

        # -------------------------
        # Tampilkan Hasil
        # -------------------------
        st.subheader(f"🔎 Prediksi: {label}")
        st.write(f"📊 Confidence: **{confidence:.2f}**")

        # Plot Probabilitas
        fig, ax = plt.subplots()
        ax.bar(labels, probs, color=["orange", "skyblue"])
        ax.set_ylabel("Probabilitas")
        ax.set_ylim(0, 1)
        for i, v in enumerate(probs):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi error saat memproses gambar: {e}")
