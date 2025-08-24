import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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

    # Separar numéricos, categóricos y fechas
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["int64", "float64", "datetime64[ns]"]).columns.tolist()

    # Detectar columnas tipo fecha (o convertibles a fecha)
    date_cols = []
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="raise")
            date_cols.append(col)
        except:
            continue

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

        # Elección de normalización
        normalize = st.checkbox("🔄 Normalizar datos con MinMaxScaler (0-1)", value=True)

        st.subheader("📦 Boxplots de variables numéricas")
        if normalize:
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
            df_melted = df_scaled.melt(var_name="Variable", value_name="Valor Normalizado")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x="Variable", y="Valor Normalizado", data=df_melted, ax=ax, palette="Set2")
            ax.set_title("Boxplots normalizados (escala 0-1)", fontsize=12)
            st.pyplot(fig)
        else:
            df_melted = df[numeric_cols].melt(var_name="Variable", value_name="Valor")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x="Variable", y="Valor", data=df_melted, ax=ax, palette="Set2")
            ax.set_title("Boxplots en escala original", fontsize=12)
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

    # --- Tendencias en el tiempo ---
    if date_cols and numeric_cols:
        st.subheader("📈 Análisis de tendencias temporales")

        date_col = st.selectbox("Seleccione la columna de fecha", date_cols)
        num_col = st.selectbox("Seleccione la variable numérica a graficar", numeric_cols)

        if date_col and num_col:
            df_sorted = df.sort_values(by=date_col)

            fig, ax = plt.subplots(figsize=(12, 5))
            sns.lineplot(x=df_sorted[date_col], y=df_sorted[num_col], ax=ax, marker="o", color="teal")
            ax.set_title(f"Tendencia temporal de {num_col} sobre {date_col}", fontsize=14)
            ax.set_xlabel("Tiempo")
            ax.set_ylabel(num_col)
            st.pyplot(fig)
