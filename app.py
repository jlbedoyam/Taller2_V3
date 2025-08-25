import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# -------------------- Función para inicializar el modelo LLM --------------------
def build_llm(hf_token: str):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
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


# -------------------- Configuración de la App --------------------
st.set_page_config(page_title="EDA + LLM", layout="wide")

st.title("📊 Exploración de Datos con EDA + 🤖 LLM")

menu = st.sidebar.radio("Navegación", ["📂 Carga de Datos", "📈 Análisis EDA", "🤖 Análisis con LLM"])


# -------------------- 📂 Carga de Datos --------------------
if menu == "📂 Carga de Datos":
    st.header("📂 Carga de Datos")

    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Convertir columnas que contienen "date" o "fecha" a formato datetime
        for col in df.columns:
            if "date" in col.lower() or "fecha" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except:
                    pass

        st.session_state.df = df
        st.success("✅ Datos cargados correctamente")

        st.subheader("Vista previa")
        st.dataframe(df.head())


# -------------------- 📈 Análisis EDA --------------------
elif menu == "📈 Análisis EDA":
    st.header("📈 Análisis Exploratorio de Datos")

    if "df" not in st.session_state:
        st.warning("Primero carga un dataset en la sección 📂 Carga de Datos")
    else:
        df = st.session_state.df

        st.subheader("Información general")
        st.write(df.describe(include="all"))

        st.subheader("Distribución de variables numéricas")
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        if len(num_cols) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            df[num_cols].hist(ax=ax)
            st.pyplot(fig)

        st.subheader("Correlaciones")
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)


# -------------------- 🤖 Análisis con LLM --------------------
elif menu == "🤖 Análisis con LLM":
    st.header("🤖 Análisis con LLM")

    if "df" not in st.session_state:
        st.warning("Primero carga un dataset en la sección 📂 Carga de Datos")
    else:
        if "hf_token" not in st.session_state:
            hf_token = st.text_input("🔑 Ingresa tu Hugging Face Token", type="password")
            if hf_token:
                st.session_state.hf_token = hf_token
                st.success("✅ Token guardado en sesión")

        if "hf_token" in st.session_state:
            llm = build_llm(st.session_state.hf_token)

            st.subheader("Haz preguntas sobre tu dataset")
            question = st.text_input("❓ Escribe tu pregunta")
            if question:
                st.write("💡 Respuesta del modelo:")
                response = llm.invoke(question)
                st.write(response)
