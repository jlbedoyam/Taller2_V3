import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import numpy as np

# 🔹 Integración con LLM
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ======================
# ESTILOS
# ======================
st.set_page_config(layout="wide", page_title="EDA Interactivo con LLM")

st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #e6f0ff;
    }
    .block-container {
        max-width: 1200px;
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
st.sidebar.title("Menú de opciones")
menu = st.sidebar.radio(
    "Selecciona una sección:",
    ["Carga de Datos", "EDA", "Asistente LLM", "📖 Documentación"]
)

# ======================
# CARGA DE DATOS
# ======================
if menu == "Carga de Datos":
    st.header("📂 Carga de Datos")

    file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("✅ Dataset cargado correctamente")
        st.dataframe(df.head())
    else:
        st.info("Por favor, carga un archivo CSV para comenzar.")

# ======================
# EDA
# ======================
elif menu == "EDA" and "df" in st.session_state:
    st.header("📊 Análisis Exploratorio de Datos")

    df = st.session_state.df

    # Información general
    st.subheader("Información general")
    st.write(df.describe(include="all"))

    # Detección de nulos
    st.subheader("Valores nulos")
    st.write(df.isnull().sum())

    # Normalización y Z-Score
    st.subheader("Transformaciones")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        option = st.selectbox("Selecciona una columna numérica:", numeric_cols)

        if option:
            col_data = df[option].dropna()

            # Normalización Min-Max
            scaler = MinMaxScaler()
            norm_data = scaler.fit_transform(col_data.values.reshape(-1, 1))

            # Z-Score
            z_scores = zscore(col_data)

            st.write(f"📌 **Columna seleccionada:** {option}")
            st.write("🔹 Normalización Min-Max (primeros 5 valores):", norm_data[:5].flatten())
            st.write("🔹 Z-Scores (primeros 5 valores):", z_scores[:5])

            # Gráfico
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(col_data, ax=ax[0], kde=True)
            ax[0].set_title("Distribución original")
            sns.histplot(norm_data.flatten(), ax=ax[1], kde=True)
            ax[1].set_title("Distribución normalizada")
            st.pyplot(fig)
    else:
        st.warning("⚠️ No hay columnas numéricas disponibles.")

# ======================
# ASISTENTE LLM
# ======================
elif menu == "Asistente LLM" and "df" in st.session_state:
    st.header("🤖 Asistente LLM sobre tu dataset")

    # --- Persistencia de API Key en la sesión ---
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = None

    # Si ya está guardada, no vuelve a pedirla
    if st.session_state.groq_api_key:
        st.info("🔑 API Key cargada en esta sesión.")
    else:
        groq_api_key = st.text_input("Ingresa tu API Key de Groq", type="password")
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
            st.success("✅ API Key guardada en esta sesión")

    # Botón para cerrar sesión de la API Key
    if st.session_state.groq_api_key:
        if st.button("Cerrar sesión de API Key"):
            st.session_state.groq_api_key = None
            st.info("🔒 API Key eliminada de la sesión.")

    # --- Si hay token, inicializamos el modelo ---
    if st.session_state.groq_api_key:
        model = ChatGroq(
            groq_api_key=st.session_state.groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )

        # Prompt Template
        template = """
        Eres un experto en ciencia de datos.
        Dataset cargado con las siguientes columnas: {columns}.
        Resumen estadístico:
        {stats}

        El usuario pregunta: {question}
        Responde en español de manera clara y concisa.
        """
        prompt = PromptTemplate(
            input_variables=["columns", "stats", "question"],
            template=template
        )
        chain = LLMChain(llm=model, prompt=prompt)

        # --- Input de pregunta ---
        user_question = st.text_input("Escribe tu pregunta sobre el dataset:")
        if user_question:
            df = st.session_state.df
            columns = ", ".join(df.columns)
            stats = df.describe(include="all").to_string()

            response = chain.run(columns=columns, stats=stats, question=user_question)
            st.markdown("### 📌 Respuesta del asistente:")
            st.write(response)
    else:
        st.warning("⚠️ Ingresa tu API Key para usar el asistente.")

# ======================
# DOCUMENTACIÓN
# ======================
elif menu == "📖 Documentación":
    st.header("📖 Documentación del Asistente LLM")

    st.markdown("""
    ## 🔹 Descripción
    Este asistente utiliza un **Modelo de Lenguaje de Gran Escala (LLM)** para analizar tu dataset
    y responder preguntas en español de manera clara y contextualizada.

    ## 🔹 Arquitectura del modelo
    - **Modelo:** `llama-3.3-70b-versatile`
    - **Proveedor:** [Groq](https://groq.com/) (API ultrarrápida para inferencia de LLMs).
    - **Integración:** Implementada mediante [LangChain](https://www.langchain.com/) y la clase `ChatGroq`.
    - **Prompting:** Se utiliza un `PromptTemplate` que incluye:
        - Columnas del dataset.
        - Resumen estadístico (`df.describe()`).
        - Pregunta del usuario.

    ## 🔹 Funcionamiento general
    1. El usuario carga un dataset en la aplicación.
    2. Se solicita (una sola vez por sesión) la **API Key de Groq**.
    3. El asistente genera respuestas basadas en:
        - Estructura y estadísticas del dataset.
        - La pregunta escrita por el usuario.
    4. La respuesta se muestra en lenguaje natural en la aplicación.

    ## 🔹 Buenas prácticas de uso
    - Asegúrate de que tu dataset esté limpio antes de hacer preguntas.
    - Usa preguntas claras y directas (ejemplo: *“¿Cuál es la media de la columna ventas por región?”*).
    - Ten en cuenta que el modelo **no ejecuta cálculos adicionales**, solo interpreta la información que se le pasa.

    ## 🔹 Limitaciones
    - Depende de la calidad y estructura del dataset cargado.
    - No reemplaza un análisis estadístico profundo, sino que lo complementa.
    - Requiere conexión a internet y una **API Key válida de Groq**.

    ---
    ✨ Este módulo busca hacer más intuitivo el **Análisis Exploratorio de Datos (EDA)**
    apoyándose en modelos de lenguaje de última generación.
    """)
