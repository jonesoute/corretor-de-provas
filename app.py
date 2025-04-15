import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import easyocr

# ------------------ CONFIGURAÇÕES GERAIS ------------------
st.set_page_config(page_title="Corretor de Provas", layout="centered")
st.title("📄 Corretor Automático de Provas")
st.markdown("Envie uma imagem da folha de respostas preenchida para correção.")
st.warning("**Atenção!** Tire a foto com o **celular na horizontal** para melhor leitura da imagem.")

# ------------------ ENTRADAS DO USUÁRIO ------------------
modo = st.radio("Modo de leitura da prova:", ["Automático (Contornos)", "Modelo Base (Template)"])
num_questions = st.number_input("Número de questões:", min_value=1, max_value=100, value=26)
num_options = st.number_input("Número de alternativas (A, B, C...):", min_value=2, max_value=5, value=4)

gabarito_str = st.text_input("Gabarito (ex: a,b,c,d,...):")
gabarito = [alt.strip().lower() for alt in gabarito_str.split(',') if alt.strip()]

uploaded_file = st.file_uploader("Envie a imagem da prova:", type=["jpg", "jpeg", "png"])

# ------------------ PRÉ-PROCESSAMENTO ------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# ------------------ DETECÇÃO AUTOMÁTICA (CONTORNOS) ------------------
def detect_answers_contours(thresh_img, num_questions, num_options):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    answer_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 3000:  # faixa de tamanho para círculos
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2:
                answer_contours.append((x, y, w, h))

    answer_contours = sorted(answer_contours, key=lambda b: (b[1], b[0]))  # ordena por linha, depois por coluna

    respostas = []
    blocos = [answer_contours[i:i + num_options] for i in range(0, len(answer_contours), num_options)]

    for bloco in blocos[:num_questions]:
        preenchimentos = []
        for (x, y, w, h) in bloco:
            roi = thresh_img[y:y + h, x:x + w]
            fill = cv2.countNonZero(roi)
            preenchimentos.append(fill)
        if preenchimentos:
            max_fill = max(preenchimentos)
            index = preenchimentos.index(max_fill)
            respostas.append(chr(97 + index))
        else:
            respostas.append(None)

    return respostas

# ------------------ DETECÇÃO POR POSIÇÃO FIXA (MODELO) ------------------
def detect_answers_fixed(thresh_img, num_questions, num_options):
    answers = []
    h, w = thresh_img.shape
    box_h = h // (num_questions // 5)
    box_w = w // 5
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
                selected_option = chr(97 + i)
        answers.append(selected_option)
    return answers

# ------------------ PROCESSAMENTO DA IMAGEM ------------------
if uploaded_file and len(gabarito) == num_questions:
    image = Image.open(uploaded_file)
    img = np.array(image)

    with st.spinner("Analisando a imagem..."):
        thresh = preprocess_image(img)
        if modo == "Automático (Contornos)":
            respostas = detect_answers_contours(thresh, num_questions, num_options)
        else:
            respostas = detect_answers_fixed(thresh, num_questions, num_options)

        acertos = sum([1 for a, b in zip(respostas, gabarito) if a == b])
        nota = (acertos / num_questions) * 10

    st.success("Correção finalizada! ✅")
    st.markdown(f"**Nota final:** {nota:.2f} / 10")
    st.markdown(f"**Acertos:** {acertos} de {num_questions}")

    st.subheader("Respostas do aluno:")
    for i, resp in enumerate(respostas, 1):
        g = gabarito[i - 1].upper()
        r = resp.upper() if resp else "-"
        certo = "✅" if resp == gabarito[i - 1] else "❌"
        st.write(f"Questão {i:02d}: Resposta = {r} | Gabarito = {g} {certo}")

elif uploaded_file and len(gabarito) != num_questions:
    st.error("O número de respostas no gabarito deve corresponder ao número de questões.")
