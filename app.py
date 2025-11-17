import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
    MobileNetV2
)
st.set_page_config(
    page_title="Classificador de Imagens IA",
    layout="centered"
)

st.title(" Classificador de Imagens com IA")
st.write("Envie uma imagem para que o modelo MobileNetV2 faça a análise.")

@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Selecione uma imagem...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagem enviada", use_column_width=True)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)
    preds = model.predict(img_preprocessed)
    results = decode_predictions(preds, top=3)[0]
    st.subheader(" Resultados da Classificação")

    for id_label, label_name, probability in results:
        st.write(f"**{label_name}** — {round(probability * 100, 2)}%")
