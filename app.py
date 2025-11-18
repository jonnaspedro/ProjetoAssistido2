import os
import io
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Classificador MNIST - Streamlit", layout="centered")

st.title("Classificador MNIST")
st.markdown(
    "Envie uma imagem do dígito (28×28 ou maior) — o app redimensiona e normaliza automaticamente."
)

MODEL_PATH = "model/final_CNN_model.h5"

@st.cache_resource
def get_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    model = load_model(path)
    return model

model = get_model()

st.write("DEBUG: Modelo carregado?", model is not None)

if model is None:
    st.error("❌ O arquivo final_CNN_model.h5 NÃO FOI ENCONTRADO no mesmo diretório do app.py.")
    st.stop()
# Upload
uploaded = st.file_uploader("Envie uma imagem (png/jpg/jpeg) ou arraste aqui", type=["png","jpg","jpeg"])
use_example = st.button("Testar com imagem de exemplo do MNIST")

def preprocess_image(img: Image.Image):
    img = img.convert("L")  
    extrema = img.getextrema()  
    if extrema[0] > 50 and extrema[1] > 200:
        img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr, img

if use_example:
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    sample_img = Image.fromarray(xte[0])
    arr, preview = preprocess_image(sample_img)
    st.subheader("Exemplo do MNIST (conjunto de teste)")
    st.image(preview, width=150)
    st.write(f"Rótulo verdadeiro: **{int(yte[0])}**")
    probs = model.predict(arr)[0]
    pred = int(np.argmax(probs))
    st.success(f"Predição: **{pred}**")
    st.write("Probabilidades por classe:")
    st.bar_chart(probs)
else:
    if uploaded is not None:
        try:
            image = Image.open(io.BytesIO(uploaded.read()))
        except Exception as e:
            st.error("Erro ao abrir a imagem. Tente outro arquivo.")
            st.stop()

        st.subheader("Imagem original")
        st.image(image, width=240)

        arr, preview = preprocess_image(image)
        st.subheader("Pré-processamento (28×28, grayscale)")
        st.image(preview, width=160)

        probs = model.predict(arr)[0]
        pred = int(np.argmax(probs))

        st.success(f"Predição: **{pred}**")
        st.write("Probabilidades por classe:")
        st.bar_chart(probs)

        st.write("---")
        st.write("Probabilidades detalhadas:")
        for i, p in enumerate(probs):
            st.write(f"Classe {i}: {p:.4f}")
    else:
        st.info("Envie um arquivo ou clique em 'Testar com imagem de exemplo do MNIST'.")
