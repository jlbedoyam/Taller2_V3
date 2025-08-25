import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# ================================
# Configuración de la página
# ================================
st.set_page_config(page_title="EDA con LLM", layout="wide")

# Paleta de colores por secciones
SECTION_COLORS = {
    "Carga de Datos": "#F0F8FF",
    "Análisis de Tendencia": "#FFF8DC",
    "Análisis de Correlación": "#E6E6FA",
    "Análisis con LLM": "#F5F5DC"
}

# ================================
# Funciones auxiliares
# ================================
def load_data(file):
    df = pd.read_csv(file)

    # Convertir solo las columnas con "date" o "fecha" en el nombre
    for col in df.columns:
        if "date" in col.lower() or "fecha" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def generate_eda_summary(df):
    summary = []

    # Número de filas y columnas
    summary.append(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Variables numéricas
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        summary.append(f"Columnas numéricas: {', '.join(num_cols)}")
        summary.append("Estadísticas descriptivas de las variables numéricas:")
        summary.append(str(df[num_cols].describe().to_dict()))

    # Variables categóricas
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        summary.append(f"Columnas categóricas: {', '.join(cat_cols)}")

    # Variables de fecha
    date_cols = [c for c in df.columns if "date" in c.lower() or "fecha" in c.lower()]
    if date_cols:
        summary.append(f"Columnas de fecha: {', '.join(date_cols)}")

    return "\n".join(summary)


def build_llm(hf_token, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto"
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


# ================================
# Layout principal
# ================================
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #ADD8E6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("📊 Menú de Navegación")
    section = st.radio(
        "Ir a:",
        ["Carga de Datos", "Análisis de Tendencia", "Análisis de Correlación", "Análisis con LLM"]
    )
    st.markdown("---")
    st.markdown("💡 **App desarrollada con Streamlit + Hugging Face**")

# ================================
# Lógica por secciones
# ================================
if section == "Carga de Datos":
    st.markdown(f"<div style='background-color:{SECTION_COLORS[section]};padding:20px;'>", unsafe_allow_html=True)
    st.header("📂 Carga de Datos")

    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    hf_token = st.text_input("🔑 Ingresa tu Hugging Face Token", type="password")

    if uploaded_file:
        data = load_data(uploaded_file)
        st.success("Datos cargados correctamente ✅")
        st.dataframe(data.head())

        # Guardar en sesión
        st.session_state["data"] = data
        st.session_state["eda_summary"] = generate_eda_summary(data)
        if hf_token:
            st.session_state["hf_token"] = hf_token

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "Análisis de Tendencia":
    st.markdown(f"<div style='background-color:{SECTION_COLORS[section]};padding:20px;'>", unsafe_allow_html=True)
    st.header("📈 Análisis de Tendencia")

    if "data" in st.session_state:
        data = st.session_state["data"]
        date_cols = [c for c in data.columns if "date" in c.lower() or "fecha" in c.lower()]
        num_cols = data.select_dtypes(include="number").columns.tolist()

        if date_cols and num_cols:
            date_col = st.selectbox("Selecciona la columna de fecha", date_cols)
            num_col = st.selectbox("Selecciona la variable numérica", num_cols)

            period = st.selectbox("Periodo de resumen", ["Día", "Mes", "Año"])
            rule = {"Día": "D", "Mes": "M", "Año": "Y"}[period]

            df_trend = data[[date_col, num_col]].dropna()
            df_trend = df_trend.groupby(pd.Grouper(key=date_col, freq=rule)).mean()

            st.line_chart(df_trend)
        else:
            st.warning("El dataset debe tener al menos una columna de fecha y una numérica.")
    else:
        st.warning("Primero carga un dataset en la sección 'Carga de Datos'.")

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "Análisis de Correlación":
    st.markdown(f"<div style='background-color:{SECTION_COLORS[section]};padding:20px;'>", unsafe_allow_html=True)
    st.header("🔗 Análisis de Correlación")

    if "data" in st.session_state:
        data = st.session_state["data"]
        num_cols = data.select_dtypes(include="number").columns.tolist()

        if len(num_cols) >= 2:
            var1 = st.selectbox("Variable 1", num_cols)
            var2 = st.selectbox("Variable 2", num_cols, index=1)

            normalize = st.checkbox("Normalizar con MinMaxScaler")

            data_corr = data[[var1, var2]].dropna()
            if normalize:
                scaler = MinMaxScaler()
                data_corr = pd.DataFrame(scaler.fit_transform(data_corr), columns=[var1, var2])

            if var1 == var2:
                st.error("❌ No tiene sentido correlacionar una variable consigo misma.")
            else:
                corr_value = data_corr[var1].corr(data_corr[var2], method="pearson")
                st.write(f"📌 Índice de correlación de Pearson: **{corr_value:.3f}**")

                fig, ax = plt.subplots()
                sns.scatterplot(x=var1, y=var2, data=data_corr, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("El dataset debe tener al menos dos variables numéricas.")
    else:
        st.warning("Primero carga un dataset en la sección 'Carga de Datos'.")

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "Análisis con LLM":
    st.markdown(f"<div style='background-color:{SECTION_COLORS[section]};padding:20px;'>", unsafe_allow_html=True)
    st.header("🤖 Análisis con LLM (Llama 3)")

    if "eda_summary" in st.session_state and "hf_token" in st.session_state:
        eda_summary = st.session_state["eda_summary"]
        hf_token = st.session_state["hf_token"]

        user_question = st.text_input("Haz una pregunta sobre los datos:")

        if user_question:
            with st.spinner("Generando respuesta con Llama 3..."):
                llm = build_llm(hf_token)
                prompt = f"Resumen del EDA:\n{eda_summary}\n\nPregunta: {user_question}"
                response = llm.invoke(prompt)
                st.success("Respuesta del modelo:")
                st.write(response)
    else:
        st.warning("Primero carga un dataset y proporciona tu Hugging Face Token en 'Carga de Datos'.")

    st.markdown("</div>", unsafe_allow_html=True)
