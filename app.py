import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# -----------------------------
# Función para construir el LLM
# -----------------------------
def build_llm(hf_token, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=hf_token
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# -----------------------------
# App en Streamlit
# -----------------------------
st.set_page_config(
    page_title="EDA + LLM con Llama 3",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("📊 Menú de opciones")

# Sección de carga de datos
st.sidebar.subheader("Carga de datos")
uploaded_file = st.sidebar.file_uploader("📂 Sube tu archivo CSV", type=["csv"])
hf_token = st.sidebar.text_input("🔑 Ingresa tu Hugging Face Token", type="password")

# Columna izquierda menú / derecha contenido
menu = st.sidebar.radio("Navegación", ["EDA Automático", "Análisis con LLM"])

# -----------------------------
# Manejo de datos
# -----------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Convertir solo columnas que contengan "date" o "fecha" en datetime
        for col in df.columns:
            if "date" in col.lower() or "fecha" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

        st.success("✅ Datos cargados correctamente")
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
        df = None
else:
    df = None

# -----------------------------
# Sección de EDA Automático
# -----------------------------
if menu == "EDA Automático":
    if df is not None:
        st.header("📊 Exploratory Data Analysis (EDA)")

        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())

        st.subheader("Resumen general")
        st.write(df.describe(include="all"))

        st.subheader("Tipos de datos")
        st.write(df.dtypes)

        # Gráfico de correlación solo si hay más de 1 variable numérica
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] > 1:
            st.subheader("Mapa de calor - Correlación")
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("⚠️ No hay suficientes variables numéricas para calcular correlación.")

    else:
        st.info("📂 Por favor carga un CSV para iniciar el análisis.")

# -----------------------------
# Sección de análisis con LLM
# -----------------------------
elif menu == "Análisis con LLM":
    st.header("🤖 Análisis con LLM (Llama 3)")

    if df is not None and hf_token:
        # Construcción del LLM
        try:
            llm = build_llm(hf_token)
            st.success("✅ LLM cargado correctamente")
        except Exception as e:
            st.error(f"❌ Error al inicializar el modelo: {e}")
            llm = None

        if llm is not None:
            # Resumen de los datos para dar contexto
            resumen = f"""
            Dataset con {df.shape[0]} filas y {df.shape[1]} columnas.
            Columnas: {', '.join(df.columns)}.
            Tipos: {df.dtypes.to_dict()}
            """

            st.subheader("Hazle una pregunta al modelo")
            user_q = st.text_area("✍️ Escribe tu pregunta sobre los datos:")

            if st.button("Preguntar al LLM"):
                if user_q.strip():
                    prompt = f"""
                    Basado en el siguiente resumen del dataset:

                    {resumen}

                    Responde la siguiente pregunta del usuario de forma clara y breve:
                    {user_q}
                    """

                    try:
                        response = llm.invoke(prompt)
                        st.markdown("### 📌 Respuesta del LLM")
                        st.write(response)
                    except Exception as e:
                        st.error(f"❌ Error al generar la respuesta: {e}")
                else:
                    st.warning("⚠️ Escribe una pregunta primero.")
    else:
        st.info("📂 Carga un CSV y proporciona tu Hugging Face Token para usar el LLM.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("💡 App desarrollada con ❤️ usando Streamlit, LangChain y Hugging Face")
