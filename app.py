import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

st.title("🔍 Classificador de Imagens — IA IFPE / Softex / Huawei")

st.write("Envie uma imagem para ser avaliada pelo modelo MobileNetV2.")

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))

    st.image(img, caption="Imagem enviada", use_container_width=True)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    preds = model.predict(img_preprocessed)
    decoded = decode_predictions(preds, top=3)[0]

    st.subheader("📌 Resultados da IA:")
    for pred in decoded:
        st.write(f"**{pred[1]}** — {pred[2]*100:.2f}%")
