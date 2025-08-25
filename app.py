import streamlit as st
import pandas as pd
from langchain_huggingface import HuggingFaceEndpoint

# ===============================
# Configuración de la página
# ===============================
st.set_page_config(page_title="App de Análisis", layout="wide")

# ===============================
# Inicialización de variables de sesión
# ===============================
if "df" not in st.session_state:
    st.session_state.df = None
if "hf_token" not in st.session_state:
    st.session_state.hf_token = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# ===============================
# Función para construir el modelo LLM
# ===============================
def build_llm(hf_token):
    return HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",  # puedes cambiar el modelo
        huggingfacehub_api_token=hf_token
    )

# ===============================
# Menú de navegación
# ===============================
st.sidebar.title("📊 Menú de Navegación")
menu = st.sidebar.radio("Ir a:", [
    "Carga de Datos",
    "Análisis de Tendencia",
    "Análisis de Correlación",
    "Análisis con LLM"
])

# ===============================
# Página: Carga de Datos
# ===============================
if menu == "Carga de Datos":
    st.header("📂 Carga de Datos")

    # Subida de archivo
    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ Archivo cargado exitosamente")
        st.dataframe(st.session_state.df.head())

    # Ingreso de Token Hugging Face
    token_input = st.text_input("🔑 Ingresa tu Hugging Face Token", type="password")
    if token_input:
        st.session_state.hf_token = token_input
        st.success("✅ Token guardado en la sesión")

# ===============================
# Página: Análisis de Tendencia
# ===============================
elif menu == "Análisis de Tendencia":
    st.header("📈 Análisis de Tendencia")
    if st.session_state.df is not None:
        st.line_chart(st.session_state.df.select_dtypes(include="number"))
    else:
        st.warning("⚠️ Primero carga un dataset en 'Carga de Datos'.")

# ===============================
# Página: Análisis de Correlación
# ===============================
elif menu == "Análisis de Correlación":
    st.header("📊 Análisis de Correlación")
    if st.session_state.df is not None:
        corr = st.session_state.df.corr(numeric_only=True)
        st.dataframe(corr)
        st.bar_chart(corr)
    else:
        st.warning("⚠️ Primero carga un dataset en 'Carga de Datos'.")

# ===============================
# Página: Análisis con LLM
# ===============================
elif menu == "Análisis con LLM":
    st.header("🤖 Análisis con LLM")

    if st.session_state.df is not None and st.session_state.hf_token:
        # Cargar modelo solo una vez
        if st.session_state.llm is None:
            with st.spinner("Cargando modelo LLaMA desde Hugging Face..."):
                st.session_state.llm = build_llm(st.session_state.hf_token)

        # Entrada de usuario
        user_query = st.text_area("Escribe tu consulta sobre los datos")
        if st.button("Analizar con LLM") and user_query:
            prompt = f"""
            Dataset columnas: {', '.join(st.session_state.df.columns)}.
            Responde en español de forma clara: {user_query}
            """
            response = st.session_state.llm.invoke(prompt)
            st.write("### Respuesta del LLM:")
            st.write(response.content)
    else:
        st.warning("⚠️ Primero carga un dataset y proporciona tu Hugging Face Token en 'Carga de Datos'.")
