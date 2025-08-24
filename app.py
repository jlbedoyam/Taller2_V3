import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA Automático", layout="wide")

st.title("📊 Aplicación de EDA Automático")

# --- Subida de archivo ---
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

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

        # Boxplots
        st.subheader("📦 Boxplots de variables numéricas")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot de {col}")
            st.pyplot(fig)

    # --- Variables categóricas ---
    if categorical_cols:
        st.subheader("📊 Frecuencias de variables categóricas")
        for col in categorical_cols:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Frecuencia de {col}")
            st.pyplot(fig)

    # --- Análisis de correlación ---
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Correlación entre dos variables numéricas")
        col1 = st.selectbox("Seleccione la primera variable", numeric_cols)
        col2 = st.selectbox("Seleccione la segunda variable", numeric_cols)

        if col1 and col2:
            corr_value = df[col1].corr(df[col2])
            st.write(f"**Coeficiente de correlación de Pearson entre {col1} y {col2}:** {corr_value:.4f}")

            # Gráfico de dispersión
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
            ax.set_title(f"Dispersión entre {col1} y {col2}")
            st.pyplot(fig)

