import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import numpy as np

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ======================
# ESTILOS
# ======================
st.set_page_config(layout="wide", page_title="EDA Interactivo + LLM")

st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #e6f0ff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 EDA Interactivo con Asistente LLM")

# ======================
# CARGA DE DATOS
# ======================
uploaded_file = st.file_uploader("Carga tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df  # Guardamos en sesión

    st.subheader("👀 Vista previa del dataset")
    st.dataframe(df.head())

    st.subheader("📌 Resumen estadístico")
    st.write(df.describe(include="all"))

    # ======================
    # Análisis Gráfico
    # ======================
    st.subheader("📈 Histogramas")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        col = st.selectbox("Selecciona una columna numérica:", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.info("⚠️ No hay columnas numéricas para graficar.")

# ======================
# ASISTENTE LLM
# ======================
st.sidebar.header("🤖 Asistente LLM")

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

if not st.session_state.groq_api_key:
    groq_api_key = st.sidebar.text_input("🔑 Ingresa tu API Key de Groq", type="password")
    if groq_api_key:
        st.session_state.groq_api_key = groq_api_key
        st.sidebar.success("✅ API Key guardada en esta sesión")
else:
    st.sidebar.info("🔑 Usando la API Key almacenada en la sesión")

    if "df" in st.session_state:
        st.header("🤖 Haz preguntas sobre tu dataset")

        model = ChatGroq(
            groq_api_key=st.session_state.groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )

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

        user_question = st.text_input("✍️ Escribe tu pregunta sobre el dataset:")
        if user_question:
            df = st.session_state.df
            columns = ", ".join(df.columns)
            stats = df.describe(include="all").to_string()

            response = chain.run(columns=columns, stats=stats, question=user_question)
            st.markdown("### 📌 Respuesta del asistente:")
            st.write(response)
    else:
        st.info("⚠️ Primero carga un dataset para poder hacer preguntas.")
