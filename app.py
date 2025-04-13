import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import easyocr

# Função para converter imagem para escala de cinza e binarizar
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# Função para reconhecer marcações (círculos preenchidos)
def detect_answers(thresh_img, num_questions, num_options):
    answers = []
    h, w = thresh_img.shape
    box_h = h // (num_questions // 5)  # ajusta conforme linhas
    box_w = w // 5  # colunas de questões
    for q in range(num_questions):
        row = q // 5
        col = q % 5
        x = col * box_w
        y = row * box_h
        roi = thresh_img[y:y + box_h, x:x + box_w]
        roi_h, roi_w = roi.shape
        opt_w = roi_w // num_options
        max_fill = 0
        selected_option = None
        for i in range(num_options):
            opt_x = i * opt_w
            opt_roi = roi[:, opt_x:opt_x + opt_w]
            fill = cv2.countNonZero(opt_roi)
            if fill > max_fill:
                max_fill = fill
                selected_option = chr(97 + i)  # a, b, c, d, e
        answers.append(selected_option)
    return answers

# Streamlit App
st.set_page_config(page_title="Corretor de Provas", layout="centered")
st.title("📄 Corretor Automático de Provas")

st.markdown("Envie uma imagem da folha de respostas preenchida para correção.")

num_questions = st.number_input("Número de questões:", min_value=1, max_value=100, value=25)
num_options = st.number_input("Número de alternativas (A, B, C...):", min_value=2, max_value=5, value=4)

gabarito_str = st.text_input("Gabarito (ex: a,b,c,d,...):")
gabarito = [alt.strip().lower() for alt in gabarito_str.split(',') if alt.strip()]

uploaded_file = st.file_uploader("Envie a imagem da prova:", type=["jpg", "jpeg", "png"])

if uploaded_file and len(gabarito) == num_questions:
    image = Image.open(uploaded_file)
    img = np.array(image)

    with st.spinner("Analisando a imagem..."):
        thresh = preprocess_image(img)
        respostas = detect_answers(thresh, num_questions, num_options)
        acertos = sum([1 for a, b in zip(respostas, gabarito) if a == b])
        nota = (acertos / num_questions) * 10

    st.success(f"Correção finalizada! ✅")
    st.markdown(f"**Nota final:** {nota:.2f} / 10")
    st.markdown(f"**Acertos:** {acertos} de {num_questions}")

    st.subheader("Respostas do aluno:")
    for i, resp in enumerate(respostas, 1):
        certo = "✅" if resp == gabarito[i-1] else "❌"
        st.write(f"Questão {i:02d}: Resposta = {resp.upper()} | Gabarito = {gabarito[i-1].upper()} {certo}")

elif uploaded_file and len(gabarito) != num_questions:
    st.error("O número de respostas no gabarito deve corresponder ao número de questões.")
