import streamlit as st
import numpy as np
import cv2
import easyocr
from PIL import Image

st.set_page_config(page_title="Leitor de Gabarito", layout="centered")

# ---------- Funções ----------

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def corrige_perspectiva(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnts = approx
            break
    else:
        return img  # se não encontrar contorno de 4 lados, retorna a imagem original

    pts = doc_cnts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped

def detect_answers_contours(thresh_img, num_questions, num_options, original_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    answer_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2:
                answer_contours.append((x, y, w, h))
                cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    answer_contours = sorted(answer_contours, key=lambda b: (b[1], b[0]))

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

    return respostas, original_img

def detect_answers_ocr(img, num_questions):
    reader = easyocr.Reader(['pt'], gpu=False)
    result = reader.readtext(img)

    answers = [None] * num_questions
    for _, text, _ in result:
        text = text.strip().lower()
        if len(text) == 2 and text[0].isdigit() and text[1].isalpha():
            idx = int(text[0]) - 1
            answers[idx] = text[1]
    return answers

def calcula_acertos(respostas, gabarito):
    acertos = 0
    for resp, gab in zip(respostas, gabarito):
        if resp == gab:
            acertos += 1
    return acertos

# ---------- Interface Streamlit ----------

st.title("📝 Leitor de Gabarito")
st.markdown("Envie uma imagem de um gabarito preenchido para leitura automática.")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

modo = st.selectbox("Modo de leitura", ["Automático (Contornos)", "OCR (Reconhecimento de Texto)"])

num_questions = st.slider("Número de questões", 1, 50, 10)
num_options = st.slider("Número de alternativas por questão", 2, 5, 4)
gabarito = st.text_input("Gabarito (ex: abcdabcdab)", max_chars=num_questions).lower()

if uploaded_file is not None and gabarito and len(gabarito) == num_questions:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = corrige_perspectiva(img)  # Correção de perspectiva

    thresh = preprocess_image(img)

    if modo == "Automático (Contornos)":
        respostas, contornos_img = detect_answers_contours(thresh, num_questions, num_options, img.copy())
    else:
        respostas = detect_answers_ocr(img, num_questions)
        contornos_img = img  # não altera a imagem

    acertos = calcula_acertos(respostas, list(gabarito))

    st.success(f"✅ Acertos: {acertos}/{num_questions}")
    st.write("Respostas detectadas:", respostas)

    st.subheader("Imagem corrigida com contornos detectados:")
    st.image(contornos_img, caption="Círculos detectados", use_column_width=True)

elif uploaded_file and gabarito and len(gabarito) != num_questions:
    st.error("❌ O número de caracteres no gabarito não corresponde ao número de questões.")
