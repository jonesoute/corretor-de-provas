[project-root]
├── app.py
├── gabarito_base.json
├── .streamlit
│   └── secrets.toml

# Conteúdo do arquivo .streamlit/secrets.toml:
# Arquivo de segredos (vazio por enquanto)

# Código principal do app:

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="Corretor de Provas", layout="wide")
st.title("📸 Correção de Provas por Imagem")

st.sidebar.header("Menu")
menu = st.sidebar.radio("Escolha uma opção:", ["1️⃣ Enviar Gabarito Base", "2️⃣ Corrigir Provas", "3️⃣ Corrigir Várias Provas"])

def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

def draw_clicks(image, clicks, color=(255, 0, 0)):
    for point in clicks:
        cv2.circle(image, (point[0], point[1]), 10, color, 3)
    return image

def draw_result_marks(image, gabarito, respostas_detectadas):
    image_result = image.copy()
    for q, ref_point in gabarito.items():
        color = (0, 255, 0) if q in respostas_detectadas else (0, 0, 255)
        cv2.circle(image_result, (ref_point[0], ref_point[1]), 10, color, 3)
    return image_result

def detect_filled_circles(image, reference_dict, threshold=0.6):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=25)

    detected_answers = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for question, ref in reference_dict.items():
            ref_point = np.array(ref)
            for (x, y, r) in circles:
                roi = gray[y - r:y + r, x - r:x + r]
                if roi.size == 0:
                    continue
                mean_intensity = np.mean(roi)
                if mean_intensity < threshold * 255:
                    distance = np.linalg.norm(np.array([x, y]) - ref_point)
                    if distance < 20:
                        detected_answers.append(question)
                        break
    return detected_answers

if menu == "1️⃣ Enviar Gabarito Base":
    st.subheader("Etapa 1: Envie a imagem do gabarito limpo")
    image_file = st.file_uploader("Faça upload de uma imagem", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        image_np = load_image(image_file)
        st.image(image_np, caption="Imagem carregada", use_column_width=True)

        st.subheader("Etapa 2: Clique nos locais onde ficam as alternativas")
        clicks = {}
        if os.path.exists("gabarito_base.json"):
            with open("gabarito_base.json", "r") as f:
                clicks = json.load(f)

        if "clicks" not in st.session_state:
            st.session_state.clicks = clicks

        st.info("Clique sobre a imagem exibida para marcar as alternativas corretas. Clique para adicionar/corrigir.")

        if st.button("Limpar marcações"):
            st.session_state.clicks = {}

        def image_click_callback(x, y):
            questao = int(st.number_input("Número da questão", min_value=1, step=1, key=f"q_{x}_{y}"))
            st.session_state.clicks[str(questao)] = [x, y]

        image_copy = image_np.copy()
        image_copy = draw_clicks(image_copy, list(st.session_state.clicks.values()))
        st.image(image_copy, caption="Imagem com marcações", use_column_width=True)

        st.write("**Coordenadas marcadas:**")
        st.json(st.session_state.clicks)

        if st.button("Salvar marcações"):
            with open("gabarito_base.json", "w") as f:
                safe_clicks = {k: [int(v[0]), int(v[1])] for k, v in st.session_state.clicks.items()}
                json.dump(safe_clicks, f)
            st.success("Marcações salvas com sucesso!")

elif menu == "2️⃣ Corrigir Provas":
    st.subheader("Etapa 3: Envie uma imagem preenchida para correção")
    prova_file = st.file_uploader("Upload da prova preenchida", type=["jpg", "png", "jpeg"], key="prova")

    if prova_file is not None:
        if not os.path.exists("gabarito_base.json"):
            st.error("Você precisa enviar e marcar o gabarito base primeiro!")
        else:
            with open("gabarito_base.json", "r") as f:
                gabarito_raw = json.load(f)
                gabarito = {int(k): v for k, v in gabarito_raw.items()}

            prova_np = load_image(prova_file)
            prova_cv2 = cv2.cvtColor(prova_np, cv2.COLOR_RGB2BGR)

            respostas_detectadas = detect_filled_circles(prova_cv2, gabarito)
            total_questoes = len(gabarito)
            acertos = sum(1 for r in respostas_detectadas if r in gabarito)

            imagem_marcada = draw_result_marks(prova_np.copy(), gabarito, respostas_detectadas)
            st.image(imagem_marcada, caption="Respostas detectadas", use_column_width=True)

            if respostas_detectadas:
                st.success(f"✅ Acertos: {acertos} / {total_questoes}")
                for q in respostas_detectadas:
                    st.write(f"Questão {q}: marcada corretamente")
            else:
                st.warning("Nenhuma resposta detectada. Verifique a qualidade da imagem ou o gabarito base.")

elif menu == "3️⃣ Corrigir Várias Provas":
    st.subheader("Etapa 3: Envie várias imagens de provas preenchidas")
    prova_files = st.file_uploader("Upload múltiplo de provas", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if prova_files:
        if not os.path.exists("gabarito_base.json"):
            st.error("Você precisa enviar e marcar o gabarito base primeiro!")
        else:
            with open("gabarito_base.json", "r") as f:
                gabarito_raw = json.load(f)
                gabarito = {int(k): v for k, v in gabarito_raw.items()}

            resultados = []
            for idx, prova_file in enumerate(prova_files, start=1):
                st.markdown(f"### 📄 Corrigindo prova #{idx}: `{prova_file.name}`")
                prova_np = load_image(prova_file)
                prova_cv2 = cv2.cvtColor(prova_np, cv2.COLOR_RGB2BGR)

                respostas_detectadas = detect_filled_circles(prova_cv2, gabarito)
                total_questoes = len(gabarito)
                acertos = sum(1 for r in respostas_detectadas if r in gabarito)

                imagem_marcada = draw_result_marks(prova_np.copy(), gabarito, respostas_detectadas)
                st.image(imagem_marcada, caption="Respostas detectadas", use_column_width=True)

                if respostas_detectadas:
                    st.success(f"✅ Acertos: {acertos} / {total_questoes}")
                else:
                    st.warning("Nenhuma resposta detectada nesta imagem.")

                resultados.append({
                    "Prova": idx,
                    "Arquivo": prova_file.name,
                    "Acertos": acertos,
                    "Total de Questões": total_questoes
                })

                st.markdown("---")

            st.markdown("## 📊 Resultado Final")
            st.dataframe(resultados, hide_index=True, use_container_width=True)
