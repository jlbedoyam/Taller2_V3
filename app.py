import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Configuración inicial ---
st.set_page_config(page_title="EDA Automático", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1, h2, h3 {color: #2c3e50;}
    .stDataFrame {background-color: white; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Explorador Automático de Datos (EDA)")

# --- Subida de archivo ---
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("👀 Vista previa de los datos")
    st.dataframe(df.head())

    # --- Detección y conversión de tipos ---
    conversion_log = []

    # Intentar convertir columnas que parezcan fechas
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
                conversion_log.append(f"📅 La columna **'{col}'** fue detectada y convertida a **fecha**.")
            except (ValueError, TypeError):
                pass  # no era fecha

    # Forzar conversión de numéricos (ej: valores con comas o espacios)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
                if df[col].notnull().sum() > 0:
                    conversion_log.append(f"🔢 La columna **'{col}'** fue convertida a **numérica**.")
            except:
                pass

    # Separar columnas por tipo
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    st.write("### 🗂️ Clasificación de variables detectadas")
    st.write(f"**Numéricas:** {numeric_cols}")
    st.write(f"**Categóricas:** {cat_cols}")
    st.write(f"**Fechas:** {date_cols}")

    # Mostrar log de conversiones en expander
    if conversion_log:
        with st.expander("🔎 Conversión automática realizada en los datos"):
            for log in conversion_log:
                st.markdown(f"- {log}")

    # --- Valores nulos y atípicos ---
    st.subheader("📉 Resumen de calidad de datos")
    nulls = df.isnull().sum()
    outliers = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers[col] = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()

    summary_quality = pd.DataFrame({
        "Nulos": nulls,
        "Atípicos (IQR)": pd.Series(outliers)
    }).fillna("-")

    st.dataframe(summary_quality)

    # --- Medidas de tendencia central ---
    st.subheader("📌 Estadísticas descriptivas (numéricas)")
    st.write(df[numeric_cols].describe())

    # --- Boxplots normalizados opcional ---
    st.subheader("📦 Distribución de variables numéricas")
    normalize = st.checkbox("🔄 Normalizar con MinMaxScaler antes del boxplot")

    data_plot = df[numeric_cols].copy()
    if normalize and not data_plot.empty:
        scaler = MinMaxScaler()
        data_plot[numeric_cols] = scaler.fit_transform(data_plot[numeric_cols])
        st.info("Los datos fueron normalizados con MinMaxScaler")

    if not data_plot.empty:
        fig, ax = plt.subplot
