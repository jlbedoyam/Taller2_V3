import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# Configuración general
# -------------------------------
st.set_page_config(
    page_title="Taller 2 - Análisis de Datos con LLM",
    layout="wide",
)

# -------------------------------
# Función para construir el LLM
# -------------------------------
def build_llm(hf_token: str):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9,
    )

    return HuggingFacePipeline(pipeline=pipe)

# -------------------------------
# Sidebar - Menú de navegación
# -------------------------------
st.sidebar.title("📌 Menú Principal")
menu = st.sidebar.radio(
    "Navegación",
    ["📂 Cargar Datos", "📊 Análisis Exploratorio", "🤖 Análisis con LLM"]
)

# -------------------------------
# Guardar token en session_state
# -------------------------------
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""

hf_token_input = st.sidebar.text_input("🔑 Ingresa tu Hugging Face Token", type="password")

# Si el usuario escribe un token nuevo, lo guardamos
if hf_token_input:
    st.session_state.hf_token = hf_token_input

# -------------------------------
# Cargar Datos
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if menu == "📂 Cargar Datos":
    st.header("📂 Cargar Datos CSV")

    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Identificar columnas de fecha (solo si contienen "date" o "fecha" en el nombre)
            date_cols = [col for col in df.columns if "date" in col.lower() or "fecha" in col.lower()]
            for col in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

            st.session_state.df = df
            st.success("✅ Datos cargados correctamente")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")

# -------------------------------
# Análisis Exploratorio
# -------------------------------
elif menu == "📊 Análisis Exploratorio":
    st.header("📊 Análisis Exploratorio de Datos (EDA)")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("📋 Información General")
        st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        st.write("Tipos de datos:")
        st.write(df.dtypes)

        st.subheader("📉 Valores nulos")
        st.write(df.isnull().sum())

        st.subheader("📈 Estadísticas descriptivas")
        st.write(df.describe(include="all"))

        # Visualización
        st.subheader("📊 Histogramas de variables numéricas")
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 0:
            fig, axes = plt.subplots(len(num_cols), 1, figsize=(8, 4 * len(num_cols)))
            if len(num_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, num_cols):
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="skyblue")
                ax.set_title(f"Histograma de {col}")
            st.pyplot(fig)
        else:
            st.info("No hay columnas numéricas para graficar.")

    else:
        st.warning("⚠️ Primero carga un dataset en la sección '📂 Cargar Datos'.")

# -------------------------------
# Análisis con LLM
# -------------------------------
elif menu == "🤖 Análisis con LLM":
    st.header("🤖 Análisis con LLM (Llama 3)")

    if st.session_state.df is not None and st.session_state.hf_token:
        df = st.session_state.df

        # Generar resumen simple del EDA para dar contexto al LLM
        resumen = f"""
        Este dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.
        Columnas: {list(df.columns)}.
        """

        # Inicializar LLM
        try:
            llm = build_llm(st.session_state.hf_token)
            st.success("✅ LLM cargado correctamente")

            # Caja de preguntas
            pregunta = st.text_input("Escribe tu pregunta sobre los datos:")
            if st.button("Preguntar al LLM"):
                if pregunta:
                    prompt = f"""
                    El usuario tiene un dataset con el siguiente resumen:
                    {resumen}

                    Responde a la siguiente pregunta en español, siendo claro y conciso:
                    {pregunta}
                    """
                    respuesta = llm.invoke(prompt)
                    st.subheader("💡 Respuesta del LLM")
                    st.write(respuesta)
        except Exception as e:
            st.error(f"❌ Error al inicializar el modelo: {e}")

    else:
        st.warning("⚠️ Debes cargar un dataset y escribir tu token de Hugging Face.")
