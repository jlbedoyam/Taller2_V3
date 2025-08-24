import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# CONFIGURACIÓN DE LA APP
# ---------------------------
st.set_page_config(page_title="EDA Automático", layout="wide")

# Estilos CSS personalizados
st.markdown(
    """
    <style>
        body {
            background-color: #F8F9FA;
        }
        .main {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 12px;
        }
        h1 {
            color: #2C3E50;
        }
        h2, h3, h4 {
            color: #34495E;
            margin-top: 30px;
        }
        .dataframe {
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .stPlotlyChart, .stPyplot {
            margin-bottom: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Aplicación de EDA Automático")

# ---------------------------
# SUBIDA DE ARCHIVO
# ---------------------------
uploaded_file = st.file_uploader("📂 Cargar archivo CSV", type=["csv"])

if uploaded_file:
    # Cargar dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Vista previa de los datos")
    st.dataframe(df.head())

    # --- Tipos de datos ---
    st.subheader("🔍 Tipos de datos")
    dtypes_df = pd.DataFrame({
        "Columna": df.columns,
        "Tipo": df.dtypes.astype(str)
    })
    st.dataframe(dtypes_df)

    # Separar numéricos y categóricos
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # --- Resumen de Nulos y Outliers ---
    st.subheader("⚠️ Resumen de valores nulos y atípicos")

    nulls = df.isnull().sum()
    nulls_pct = (nulls / len(df)) * 100

    outlier_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = outliers

    resumen = pd.DataFrame({
        "Nulos": nulls,
        "Porcentaje Nulos (%)": nulls_pct.round(2),
        "Atípicos (solo numéricas)": [outlier_summary.get(col, "-") for col in df.columns]
    })

    st.dataframe(resumen)

    # --- Estadísticos numéricos ---
    if numeric_cols:
        st.subheader("📈 Estadísticas de variables numéricas")
        st.write(df[numeric_cols].describe())

        # Boxplots agrupados
        st.subheader("📦 Boxplots de variables numéricas (agrupados)")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_melted = df[numeric_cols].melt(var_name="Variable", value_name="Valor")
        sns.boxplot(x="Variable", y="Valor", data=df_melted, ax=ax, palette="Set2")
        ax.set_title("Boxplots comparativos de variables numéricas", fontsize=12)
        st.pyplot(fig)

    # --- Variables categóricas ---
    if categorical_cols:
        st.subheader("📊 Frecuencias de variables categóricas")
        for col in categorical_cols:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind="bar", ax=ax, color="coral")
            ax.set_title(f"Frecuencia de {col}", fontsize=12)
            st.pyplot(fig)

    # --- Análisis de correlación ---
    if len(numeric_cols) >= 2:
        st.subheader("🧩 Matriz de correlación (Heatmap)
