import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2

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

# Função para aplicar filtros de ruído
def apply_noise_filter(image, filter_type, kernel_size):
    # Converte PIL para OpenCV (numpy array)
    img_array = np.array(image)

    # Se imagem for em escala de cinza, mantém 2D
    if len(img_array.shape) == 2:
        pass
    else:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if filter_type == "Média (Blur)":
        filtered = cv2.blur(img_array, (kernel_size, kernel_size))
    elif filter_type == "Gaussian Blur":
        filtered = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    elif filter_type == "Mediana":
        filtered = cv2.medianBlur(img_array, kernel_size)
    elif filter_type == "Bilateral":
        filtered = cv2.bilateralFilter(img_array, d=kernel_size, sigmaColor=75, sigmaSpace=75)
    else:
        return image

    # Converte de volta para PIL
    if len(filtered.shape) == 2:
        return Image.fromarray(filtered)
    else:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        return Image.fromarray(filtered)

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

# --- NOVO: Tipos de filtragem de ruído ---
filter_type = st.sidebar.selectbox(
    "Tipo de Filtro de Ruído",
    ["Nenhum", "Média (Blur)", "Gaussian Blur", "Mediana", "Bilateral"]
)
kernel_size = st.sidebar.slider("Tamanho do Kernel (ímpar)", 1, 15, 3, step=2)

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

    # Aplica filtro de ruído, se selecionado
    if filter_type != "Nenhum":
        image = apply_noise_filter(image, filter_type, kernel_size)

    # Exibe imagem processada
    st.image(image, caption="Imagem Processada", use_column_width=True)
else:
    st.write("Por favor, carregue uma imagem para começar.")
