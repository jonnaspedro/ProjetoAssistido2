import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ================================
# Configuração da página
# ================================
st.set_page_config(page_title="Classificador MNIST", page_icon="🧠")

st.title("🧠 Classificador de Dígitos MNIST")
st.write("Envie uma imagem 28x28 (preta e branca) para ser classificada pelo modelo treinado.")

# ================================
# Carregar modelo
# ================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_CNN_model.h5")
    return model

model = load_model()

# ================================
# Input de imagem do usuário
# ================================
uploaded_file = st.file_uploader("Envie uma imagem:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Converte para preto e branco
    st.image(image, caption="Imagem enviada", width=200)

    # Preprocessamento
    img = image.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predição
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.subheader(f"🔢 Dígito identificado: **{digit}**")

    st.bar_chart(prediction[0])
