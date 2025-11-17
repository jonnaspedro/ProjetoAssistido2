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

st.title("🧠 Classificador de Imagens com IA")
st.write("Envie uma imagem para que o modelo MobileNetV2 faça a classificação.")

@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

uploaded_file = st.file_uploader("Selecione uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    placeholder = st.empty()

    with placeholder.container():
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Imagem enviada", use_column_width=True)

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        results = decode_predictions(preds, top=3)[0]

        st.subheader("🔍 Resultados:")
        for _, label, prob in results:
            st.write(f"{label} — {round(prob * 100, 2)}%")
