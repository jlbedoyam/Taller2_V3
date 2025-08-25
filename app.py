# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --------------------------------
# Función para construir el LLM
# --------------------------------
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

# --------------------------------
# Configuración UI
# --------------------------------
st.set_page_config(page_title="App de Análisis", layout="wide")
st.sidebar.header("📊 Menú de Navegación")
menu = st.sidebar.radio("Ir a:", ["Carga de Datos", "Análisis de Tendencia", "Análisis de Correlación", "Análisis con LLM"])

# Inicializar variables de sesión
if "df" not in st.session_state:
    st.session_state.df = None
if "hf_token" not in st.session_state:
    st.session_state.hf_token = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# --------------------------------
# Página 1: Carga de Datos
# --------------------------------
if menu == "Carga de Datos":
    st.subheader("📂 Carga de Datos")

    file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    token_input = st.text_input("🔑 Ingresa tu Hugging Face Token", type="password")

    if token_input:
        st.session_state.hf_token = token_input
        st.success("✅ Token guardado en la sesión")

    if file:
        df = pd.read_csv(file)

        # Convertir en fecha solo las columnas con "Date" o "fecha"
        for col in df.columns:
            if "date" in col.lower() or "fecha" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except:
                    pass  

        st.session_state.df = df
        st.success("✅ Datos cargados exitosamente")
        st.dataframe(df.head())

# --------------------------------
# Página 2: Análisis de Tendencia
# --------------------------------
elif menu == "Análisis de Tendencia":
    if st.session_state.df is not None:
        st.subheader("📈 Análisis de Tendencia")
        df = st.session_state.df

        date_cols = [col for col in df.columns if "date" in col.lower() or "fecha" in col.lower()]
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if date_cols and num_cols:
            col_date = st.selectbox("Selecciona la columna de fecha", date_cols)
            col_num = st.selectbox("Selecciona la variable numérica", num_cols)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df[col_date], df[col_num])
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Se requieren columnas de fecha y numéricas")
    else:
        st.warning("Primero carga un dataset en 'Carga de Datos'.")

# --------------------------------
# Página 3: Análisis de Correlación
# --------------------------------
elif menu == "Análisis de Correlación":
    if st.session_state.df is not None:
        st.subheader("📊 Análisis de Correlación")
        df = st.session_state.df

        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Primero carga un dataset en 'Carga de Datos'.")

# --------------------------------
# Página 4: Análisis con LLM
# --------------------------------
elif menu == "Análisis con LLM":
    if st.session_state.df is not None and st.session_state.hf_token:
        st.subheader("🤖 Análisis con LLM")

        if st.session_state.llm is None:
            with st.spinner("Cargando modelo LLaMA desde Hugging Face..."):
                st.session_state.llm = build_llm(st.session_state.hf_token)

        user_query = st.text_area("Escribe tu consulta sobre los datos")
        if st.button("Analizar con LLM"):
            if user_query:
                prompt = f"""
                Dataset columnas: {', '.join(st.session_state.df.columns)}.
                Responde en español de forma clara: {user_query}
                """
                response = st.session_state.llm.invoke(prompt)
                st.write("### Respuesta del LLM:")
                st.write(response.content)
            else:
                st.warning("Escribe una consulta para el LLM")
    else:
        st.warning("Primero carga un dataset y proporciona tu Hugging Face Token en 'Carga de Datos'.")
