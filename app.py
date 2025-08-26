import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline


# --------------------------------------------------
# Función para construir el LLM con Hugging Face
# --------------------------------------------------
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


# --------------------------------------------------
# Configuración de la App
# --------------------------------------------------
st.set_page_config(page_title="App de Análisis + LLM", layout="wide")
st.sidebar.title("📊 Menú de Navegación")
menu = st.sidebar.radio("Ir a:", ["Carga de Datos", "Análisis de Tendencia", "Análisis de Correlación", "Análisis con LLM"])


# --------------------------------------------------
# 1. Carga de Datos
# --------------------------------------------------
if menu == "Carga de Datos":
    st.header("📂 Carga de Datos")

    file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    hf_token = st.text_input("🔑 Ingresa tu Hugging Face Token", type="password")

    if file is not None:
        df = pd.read_csv(file)

        # Intentar conversión de fechas
        for col in df.columns:
            if "date" in col.lower() or "fecha" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        st.session_state.df = df
        st.success("✅ Datos cargados correctamente")

        st.write("Vista previa:")
        st.dataframe(df.head())

    if hf_token:
        st.session_state.hf_token = hf_token
        st.success("🔑 Token almacenado en sesión")


# --------------------------------------------------
# 2. Análisis de Tendencia
# --------------------------------------------------
elif menu == "Análisis de Tendencia":
    st.header("📈 Análisis de Tendencia")

    if "df" not in st.session_state:
        st.warning("⚠️ Primero carga un CSV en 'Carga de Datos'")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        if not numeric_cols or not date_cols:
            st.error("No se encontraron columnas de fecha y numéricas adecuadas.")
        else:
            date_col = st.selectbox("Selecciona columna de fecha", date_cols)
            num_col = st.selectbox("Selecciona columna numérica", numeric_cols)

            fig, ax = plt.subplots()
            df.groupby(date_col)[num_col].mean().plot(ax=ax)
            plt.title(f"Tendencia de {num_col} a lo largo de {date_col}")
            st.pyplot(fig)


# --------------------------------------------------
# 3. Análisis de Correlación
# --------------------------------------------------
elif menu == "Análisis de Correlación":
    st.header("📊 Análisis de Correlación")

    if "df" not in st.session_state:
        st.warning("⚠️ Primero carga un CSV en 'Carga de Datos'")
    else:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Se necesitan al menos 2 columnas numéricas.")
        else:
            corr = df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            plt.title("Matriz de Correlación")
            st.pyplot(fig)


# --------------------------------------------------
# 4. Análisis con LLM
# --------------------------------------------------
elif menu == "Análisis con LLM":
    st.header("🤖 Análisis con LLM (Llama 3)")

    if "df" not in st.session_state:
        st.warning("⚠️ Primero carga un CSV en 'Carga de Datos'")
    elif "hf_token" not in st.session_state:
        st.warning("⚠️ Ingresa tu token de Hugging Face en 'Carga de Datos'")
    else:
        # Inicializar LLM (solo una vez)
        if "llm" not in st.session_state:
            with st.spinner("Cargando modelo Llama 3 desde Hugging Face..."):
                st.session_state.llm = build_llm(st.session_state.hf_token)

        user_query = st.text_area("Escribe tu consulta sobre los datos:")

        if st.button("Generar Respuesta"):
            df = st.session_state.df
            llm = st.session_state.llm

            # Construimos un prompt simple
            prompt = f"""
            Tengo un DataFrame con las siguientes columnas: {list(df.columns)}.
            Responde a la siguiente consulta sobre los datos:
            {user_query}
            """

            response = llm.invoke(prompt)

            st.write("### Respuesta del LLM:")
            st.write(str(response))
