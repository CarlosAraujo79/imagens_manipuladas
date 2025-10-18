import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2

# ===============================================================
# 🔧 Funções Utilitárias
# ===============================================================

def ensure_rgb(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    return img_array

def ensure_grayscale(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return img_array

# ===============================================================
# 🎨 Funções de Processamento Básico
# ===============================================================

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
    img_array = ensure_rgb(image)
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

    filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    return Image.fromarray(filtered)

# ===============================================================
# 🧠 Funções de Segmentação
# ===============================================================

def segment_threshold(image, threshold_value):
    gray = ensure_grayscale(image)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)

def segment_adaptive(image):
    gray = ensure_grayscale(image)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(adaptive)

def segment_kmeans(image, k):
    img = ensure_rgb(image)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape((img.shape))
    return Image.fromarray(segmented_image)

def segment_edges(image):
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges)

def segment_contours(image):
    gray = ensure_grayscale(image)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = ensure_rgb(image).copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
    return Image.fromarray(result)

# ===============================================================
# 🎯 Segmentação por Cor Aprimorada
# ===============================================================

def segment_color(image, selected_rgb, hue_tol=15, sat_tol=50, val_tol=50, smooth=True):
    img = ensure_rgb(image)  # garante RGB
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    color_bgr = np.uint8([[selected_rgb[::-1]]])
    hsv_color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color

    sat_tol = min(sat_tol, 255 - s)
    val_tol = min(val_tol, 255 - v)

    lower = np.array([max(0, h - hue_tol), max(0, s - sat_tol), max(0, v - val_tol)], dtype=np.uint8)
    upper = np.array([min(179, h + hue_tol), min(255, s + sat_tol), min(255, v + val_tol)], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    if smooth:
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

    segmented = cv2.bitwise_and(img, img, mask=mask)
    return Image.fromarray(segmented), Image.fromarray(mask)

# ===============================================================
# 🔍 Filtros de Detecção de Características
# ===============================================================

def feature_sobel(image):
    gray = ensure_grayscale(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    sobel = np.uint8(np.clip(magnitude, 0, 255))
    return Image.fromarray(sobel)

def feature_laplacian(image):
    gray = ensure_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.absolute(laplacian), 0, 255))
    return Image.fromarray(laplacian)

def feature_harris(image, block_size=2, ksize=3, k=0.04):
    gray = np.float32(ensure_grayscale(image))
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    result = ensure_rgb(image).copy()
    result[dst > 0.01 * dst.max()] = [255, 0, 0]
    return Image.fromarray(result)

def feature_shi_tomasi(image, max_corners=100):
    gray = ensure_grayscale(image)
    result = ensure_rgb(image).copy()

    h, w = gray.shape
    if h < 2 or w < 2:
        return Image.fromarray(result)

    try:
        corners = cv2.goodFeaturesToTrack(gray, max_corners, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corners = corners.astype(int)  # <- substitui np.int0
            for c in corners:
                x, y = c.ravel()
                cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
    except cv2.error:
        pass

    return Image.fromarray(result)


def feature_orb(image, max_features=300):
    img = ensure_rgb(image)
    gray = ensure_grayscale(image)
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints = orb.detect(gray, None)
    result = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
    return Image.fromarray(result)

# ===============================================================
# 🌈 Interface Streamlit
# ===============================================================

st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader("Escolha uma imagem", type=["jpg","jpeg","png"])

num_colors = st.sidebar.slider("Número de Cores para Quantização", 1, 256, 16)
rotation = st.sidebar.slider("Rotacionar Imagem (graus)", 0, 360, 0)
flip = st.sidebar.checkbox("Espelhar Imagem Horizontalmente")
grayscale = st.sidebar.checkbox("Converter para Escala de Cinza")

filter_type = st.sidebar.selectbox(
    "Tipo de Filtro de Ruído",
    ["Nenhum","Média (Blur)","Gaussian Blur","Mediana","Bilateral"]
)
kernel_size = st.sidebar.slider("Tamanho do Kernel (ímpar)", 1, 15, 3, step=2)

st.sidebar.subheader("Abordagens de Segmentação")
segmentation_type = st.sidebar.selectbox(
    "Tipo de Segmentação",
    ["Nenhum","Limiarização Simples","Limiarização Adaptativa","K-Means",
     "Detecção de Bordas (Canny)","Contornos","Segmentação por Cor (HSV)"]
)
threshold_value = st.sidebar.slider("Valor do Limiar", 0, 255, 127)
k_value = st.sidebar.slider("Número de Clusters (K-Means)", 2, 10, 3)

# Segmentação por Cor
if segmentation_type == "Segmentação por Cor (HSV)":
    selected_color = st.sidebar.color_picker("Escolha a cor base", "#00ff00")
    selected_rgb = tuple(int(selected_color.lstrip("#")[i:i+2],16) for i in (0,2,4))
    hue_tol = st.sidebar.slider("Tolerância Hue ±",0,60,15)
    sat_tol = st.sidebar.slider("Tolerância Saturação ±",0,128,50)
    val_tol = st.sidebar.slider("Tolerância Brilho ±",0,128,50)
    smooth = st.sidebar.checkbox("Suavizar máscara", value=True)

st.sidebar.subheader("Filtros de Detecção de Características")
feature_type = st.sidebar.selectbox(
    "Selecionar Filtro de Características",
    ["Nenhum","Sobel (Gradiente)","Laplaciano","Harris","Shi-Tomasi","ORB"]
)

# ===============================================================
# 🚀 Processamento da Imagem
# ===============================================================

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = quantize_image(image, num_colors)
    if grayscale:
        image = convert_to_grayscale(image)
    image = apply_geometric_transform(image, rotation, flip)
    if filter_type != "Nenhum":
        image = apply_noise_filter(image, filter_type, kernel_size)

    mask_image = None

    # Segmentação
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
        image, mask_image = segment_color(image, selected_rgb, hue_tol, sat_tol, val_tol, smooth)

    # Filtros de Características
    if feature_type == "Sobel (Gradiente)":
        image = feature_sobel(image)
    elif feature_type == "Laplaciano":
        image = feature_laplacian(image)
    elif feature_type == "Harris":
        image = feature_harris(image)
    elif feature_type == "Shi-Tomasi":
        image = feature_shi_tomasi(image)
    elif feature_type == "ORB":
        image = feature_orb(image)

    st.image(image, caption="Imagem Processada", use_column_width=True)
    if mask_image is not None:
        st.image(mask_image, caption="Máscara da Segmentação", use_column_width=True)
else:
    st.write("Por favor, carregue uma imagem para começar.")
