import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2

# --------------------------
# Funções existentes
# --------------------------

def quantize_image(image, num_colors):
    return image.convert('P', palette=Image.ADAPTIVE, colors=num_colors).convert('RGB')

def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

def apply_geometric_transform(image, rotation, flip):
    if flip:
        image = ImageOps.mirror(image)
    if rotation != 0:
        image = image.rotate(rotation)
    return image

def apply_noise_filter(image, filter_type, kernel_size):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
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

    if len(filtered.shape) == 3:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    return Image.fromarray(filtered)

# --------------------------
# Funções de segmentação
# --------------------------

def segment_threshold(image, threshold_value):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)

def segment_adaptive(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(adaptive)

def segment_kmeans(image, k):
    img = np.array(image)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape((img.shape))
    return Image.fromarray(segmented_image)

def segment_edges(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges)

def segment_contours(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.array(image).copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
    return Image.fromarray(result)

# --------------------------
# NOVO: Segmentação por cor
# --------------------------

def segment_color(image, lower_hsv, upper_hsv):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    segmented = cv2.bitwise_and(img, img, mask=mask)
    return Image.fromarray(segmented)

# --------------------------
# Interface Streamlit
# --------------------------

st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

num_colors = st.sidebar.slider("Número de Cores para Quantização", 1, 256, 16)
rotation = st.sidebar.slider("Rotacionar Imagem (graus)", 0, 360, 0)
flip = st.sidebar.checkbox("Espelhar Imagem Horizontalmente")
grayscale = st.sidebar.checkbox("Converter para Escala de Cinza")

filter_type = st.sidebar.selectbox(
    "Tipo de Filtro de Ruído",
    ["Nenhum", "Média (Blur)", "Gaussian Blur", "Mediana", "Bilateral"]
)
kernel_size = st.sidebar.slider("Tamanho do Kernel (ímpar)", 1, 15, 3, step=2)

# --- Segmentação ---
st.sidebar.subheader("Abordagens de Segmentação")
segmentation_type = st.sidebar.selectbox(
    "Tipo de Segmentação",
    ["Nenhum", "Limiarização Simples", "Limiarização Adaptativa", "K-Means",
     "Detecção de Bordas (Canny)", "Contornos", "Segmentação por Cor (HSV)"]
)

# Parâmetros para segmentações específicas
threshold_value = st.sidebar.slider("Valor do Limiar (para Limiarização)", 0, 255, 127)
k_value = st.sidebar.slider("Número de Clusters (para K-Means)", 2, 10, 3)

# --- NOVO: controles HSV ---
if segmentation_type == "Segmentação por Cor (HSV)":
    st.sidebar.markdown("### Intervalo HSV")
    lower_h = st.sidebar.slider("Hue (min)", 0, 179, 0)
    upper_h = st.sidebar.slider("Hue (max)", 0, 179, 179)
    lower_s = st.sidebar.slider("Saturação (min)", 0, 255, 50)
    upper_s = st.sidebar.slider("Saturação (max)", 0, 255, 255)
    lower_v = st.sidebar.slider("Valor (min)", 0, 255, 50)
    upper_v = st.sidebar.slider("Valor (max)", 0, 255, 255)

# --------------------------
# Processamento
# --------------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = quantize_image(image, num_colors)

    if grayscale:
        image = convert_to_grayscale(image)

    image = apply_geometric_transform(image, rotation, flip)

    if filter_type != "Nenhum":
        image = apply_noise_filter(image, filter_type, kernel_size)

    # --- Segmentação ---
    if segmentation_type == "Limiarização Simples":
        image = segment_threshold(image, threshold_value)
    elif segmentation_type == "Limiarização Adaptativa":
        image = segment_adaptive(image)
    elif segmentation_type == "K-Means":
        image = segment_kmeans(image, k_value)
    elif segmentation_type == "Detecção de Bordas (Canny)":
        image = segment_edges(image)
    elif segmentation_type == "Contornos":
        image = segment_contours(image)
    elif segmentation_type == "Segmentação por Cor (HSV)":
        lower = (lower_h, lower_s, lower_v)
        upper = (upper_h, upper_s, upper_v)
        image = segment_color(image, lower, upper)

    st.image(image, caption="Imagem Processada", use_column_width=True)
else:
    st.write("Por favor, carregue uma imagem para começar.")
