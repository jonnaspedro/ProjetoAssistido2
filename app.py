import streamlit as st
import numpy as np
from PIL import Image
import requests
import base64

st.set_page_config(page_title="Classificador de Imagens IA")

st.title("🧠 Classificador de Imagens IA")
st.write("Usando o modelo MobileNetV2 hospedado na Hugging Face.")

API_URL = "https://api-inference.huggingface.co/models/google/mobilenet_v2_1.0_224"

headers = {"Authorization": "Bearer hf_123456789"} 

def classify_image(img):
    buffered = st.runtime.memory_upload(img, "temp.jpg")
    with open(buffered, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

arquivo = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if arquivo:
    img = Image.open(arquivo).convert("RGB")
    st.image(img, use_column_width=True)

    resultados = classify_image(arquivo)

    st.subheader("Resultado:")
    for item in resultados:
        st.write(f"{item['label']} — {round(item['score']*100, 2)}%")
