import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Função para quantização de cores
def quantize_image(image, num_colors):
    return image.convert('P', palette=Image.ADAPTIVE, colors=num_colors).convert('RGB')

# Função para converter imagem para escala de cinza
def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

# Função para aplicar transformações geométricas
def apply_geometric_transform(image, rotation, flip):
    if flip:
        image = ImageOps.mirror(image)
    if rotation != 0:
        image = image.rotate(rotation)
    return image

# Configurações da barra lateral
st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

# Configurações de quantização de cores
num_colors = st.sidebar.slider("Número de Cores para Quantização", 1, 256, 16)

# Configurações de transformação geométrica
rotation = st.sidebar.slider("Rotacionar Imagem (graus)", 0, 360, 0)
flip = st.sidebar.checkbox("Espelhar Imagem Horizontalmente")

# Configurações de conversão de sistema de cores
grayscale = st.sidebar.checkbox("Converter para Escala de Cinza")

# Se uma imagem foi carregada, processa a imagem
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Aplica quantização de cores
    image = quantize_image(image, num_colors)

    # Converte para escala de cinza, se necessário
    if grayscale:
        image = convert_to_grayscale(image)

    # Aplica transformações geométricas
    image = apply_geometric_transform(image, rotation, flip)

    # Exibe a imagem original e a processada
    st.image(image, caption="Imagem Processada", use_column_width=True)
else:
    st.write("Por favor, carregue uma imagem para começar.")

