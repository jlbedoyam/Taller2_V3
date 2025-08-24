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
        st.subheader("🧩 Matriz de correlación (Heatmap)")

        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            corr_matrix, annot=True, cmap="RdYlGn", center=0,
            fmt=".2f", ax=ax, cbar=True
        )
        ax.set_title("Matriz de correlación (verde = +, rojo = -)", fontsize=14)
        st.pyplot(fig)

        # Selección de dos variables
        st.subheader("🔗 Correlación entre dos variables numéricas")
        col1 = st.selectbox("Seleccione la primera variable", numeric_cols)
        col2 = st.selectbox("Seleccione la segunda variable", numeric_cols)

        if col1 and col2:
            corr_value = df[col1].corr(df[col2])
            st.write(f"**Coeficiente de correlación de Pearson entre {col1} y {col2}:** `{corr_value:.4f}`")

            # Gráfico de dispersión
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col1], y=df[col2], ax=ax, color="purple", alpha=0.7)
            ax.set_title(f"Dispersión entre {col1} y {col2}", fontsize=12)
            st.pyplot(fig)
