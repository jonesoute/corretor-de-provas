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
usar_perspectiva = st.checkbox("Aplicar correção de perspectiva", value=True)

modo = st.selectbox("Modo de leitura da prova:", ["Automático (Contornos)", "Modelo Base (Template)"])
num_questions = st.number_input("Número de questões:", min_value=1, max_value=50, value=26)
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

# ------------------ CORREÇÃO DE PERSPECTIVA ------------------
def corrige_perspectiva(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            break
    else:
        return img  # Retorna imagem original se falhar

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    rect = order_points(pts)
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

# ------------------ DETECÇÃO AUTOMÁTICA (CONTORNOS) ------------------
def detect_answers_contours(thresh_img, num_questions, num_options):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    answer_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2:
                answer_contours.append((x, y, w, h))

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

    return respostas

# ------------------ DETECÇÃO POR POSIÇÃO FIXA (MODELO) ------------------
def detect_answers_fixed(thresh_img, num_questions, num_options):
    answers = []
    h, w = thresh_img.shape
    box_h = h // (num_questions // 5 + 1)
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

        if usar_perspectiva:
            img = corrige_perspectiva(img)

        st.subheader("Imagem após correção de perspectiva:")
        st.image(img, caption="Imagem corrigida", use_column_width=True)

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
