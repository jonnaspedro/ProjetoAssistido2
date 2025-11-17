import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
    MobileNetV2
)

st.set_page_config(page_title="Classificador de Imagens IA")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

st.title("🧠 Classificador de Imagens IA")

arquivo = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if arquivo:
    cont = st.container()
    with cont:
        img = Image.open(arquivo).convert("RGB")
        st.image(img, use_column_width=True)

        img2 = img.resize((224, 224))
        arr = np.array(img2, dtype=np.float32)
        arr = np.expand_dims(arr, 0)
        arr = preprocess_input(arr)

        pred = model.predict(arr)
        res = decode_predictions(pred, top=3)[0]

        st.subheader("Resultado:")
        for _, label, prob in res:
            st.write(f"{label} — {round(prob*100, 2)}%")
