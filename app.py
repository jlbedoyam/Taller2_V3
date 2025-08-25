import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

# ============================
# Función para cargar datos
# ============================
def cargar_datos(file):
    try:
        df = pd.read_csv(file)
    except:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            raise ValueError(f"Error al leer archivo: {e}")

    # Convertir columnas que contengan 'date' o 'fecha'
    for col in df.columns:
        if "date" in col.lower() or "fecha" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"No se pudo convertir {col} a fecha: {e}")
    return df

# ============================
# Función para generar EDA
# ============================
def generar_eda(df):
    resumen = {}
    resumen["shape"] = df.shape
    resumen["columnas"] = list(df.columns)
    resumen["tipos"] = df.dtypes.astype(str).to_dict()
    resumen["nulos"] = df.isnull().sum().to_dict()
    resumen["estadisticas"] = df.describe(include="all").transpose().to_dict()

    # Gráficos básicos
    st.subheader("Distribución de Variables Numéricas")
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribución de {col}")
            st.pyplot(fig)

    st.subheader("Variables Categóricas")
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            fig, ax = plt.subplots()
            df[col].value_counts().head(10).plot(kind="bar", ax=ax)
            ax.set_title(f"Top categorías de {col}")
            st.pyplot(fig)

    return resumen

# ============================
# Función para conectar LLM
# ============================
def crear_llm(hf_token):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=pipe)

# ============================
# Streamlit App
# ============================
st.title("📊 Explorador de Datos con EDA + LLM 🤖")

st.sidebar.header("Carga de datos y configuración")

# Subida de archivo
file = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
hf_token = st.sidebar.text_input("🔑 Ingresa tu Hugging Face Token", type="password")

if file is not None:
    df = cargar_datos(file)
    st.success("✅ Datos cargados correctamente")

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    st.header("📈 Análisis Exploratorio de Datos (EDA)")
    resumen = generar_eda(df)

    # Mostrar resumen textual
    st.subheader("Resumen del dataset")
    st.json(resumen)

    # ============================
    # Sección de análisis con LLM
    # ============================
    if hf_token:
        st.header("🤖 Análisis con LLM")
        st.write("Haz preguntas sobre el dataset y el modelo responderá en base al EDA.")

        # Crear LLM
        llm = crear_llm(hf_token)

        # Input del usuario
        pregunta = st.text_input("Escribe tu pregunta sobre el dataset:")

        if pregunta:
            # Generar contexto del EDA
            contexto = f"""
            El dataset tiene {resumen['shape'][0]} filas y {resumen['shape'][1]} columnas.
            Columnas: {resumen['columnas']}
            Tipos de variables: {resumen['tipos']}
            Valores nulos: {resumen['nulos']}
            """

            prompt = f"Contexto del dataset:\n{contexto}\n\nPregunta del usuario: {pregunta}\n\nRespuesta:"
            respuesta = llm.invoke(prompt)
            st.subheader("Respuesta del LLM")
            st.write(respuesta)
    else:
        st.info("⚠️ Ingresa tu Hugging Face Token en la barra lateral para usar el LLM.")
