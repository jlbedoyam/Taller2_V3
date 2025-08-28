import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import numpy as np

# 🔹 Integración con LLM
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ======================
# ESTILOS / PAGE CONFIG
# ======================
st.set_page_config(layout="wide", page_title="EDA Interactivo con LLM")

st.markdown("""
    <style>
    .sidebar .sidebar-content { background-color: #e6f0ff; }
    .block-container { max-width: 1300px; padding: 1.2rem 2rem; }
    h1, h2, h3 { color: #003366; }
    </style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio(
    "📊 Menú de navegación",
    [
        "Carga de datos",
        "Descripción general",
        "Análisis de valores nulos y atípicos",
        "Visualización numérica",
        "Visualización categórica",
        "Correlaciones",
        "Análisis de tendencias",
        "Pivot Table",
        "Asistente LLM",
        "📖 Documentación"
    ]
)

# ======================
# SESSION STATE
# ======================
if "df" not in st.session_state:
    st.session_state.df = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = None

st.sidebar.markdown("---")
st.sidebar.info("App desarrollada con ❤️ usando Streamlit")

# ======================
# CARGA DE DATOS
# ======================
if menu == "Carga de datos":
    st.header("📂 Carga de datos")
    file = st.file_uploader("Sube tu archivo CSV", type="csv")

    if file:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1")

        # --- Detección y conversión de fechas ---
        for col in df.columns:
            if "date" in col.lower() or "fecha" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    st.write(f"✅ Columna '{col}' convertida a fecha.")
                except Exception as e:
                    st.warning(f"⚠️ No se pudo convertir '{col}' a fecha: {e}")

        # --- Conversión de objetos a numérico si aplica ---
        for col in df.columns:
            if df[col].dtype == "object":
                tmp = pd.to_numeric(df[col], errors="coerce")
                if (tmp.notnull().sum() / len(df) > 0.9) and df[col].nunique() > 10:
                    df[col] = tmp
                    st.write(f"✅ Columna '{col}' convertida a numérico.")

        st.session_state.df = df
        st.success("✅ Datos cargados correctamente")
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.info("Carga un CSV para comenzar.")

# ======================
# DESCRIPCIÓN GENERAL
# ======================
elif menu == "Descripción general" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📖 Descripción general")
    st.subheader("Tipos de datos")
    st.dataframe(df.dtypes, use_container_width=True)

    st.subheader("Resumen estadístico (numéricas)")
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        st.write(df[num_cols].describe())
    else:
        st.warning("No se detectaron variables numéricas.")

# ======================
# ANÁLISIS DE VALORES NULOS Y ATÍPICOS
# ======================
elif menu == "Análisis de valores nulos y atípicos" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("🔎 Valores nulos y atípicos")

    st.subheader("Valores nulos por columna")
    st.write(df.isnull().sum())
