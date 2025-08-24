import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="EDA Automático", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Exploratory Data Analysis Automático")

# --- Cargar archivo ---
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Vista previa de los datos")
    st.dataframe(df.head())

    # --- Detección y conversión de tipos ---
    conversion_log = []

    for col in df.columns:
        if df[col].dtype == "object":
            # 1. Intentar convertir a fecha
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
                conversion_log.append(f"📅 La columna **'{col}'** fue detectada y convertida a **fecha**.")
                continue
            except (ValueError, TypeError):
                pass

            # 2. Intentar convertir a numérico SOLO si mayoría son números
            numeric_attempt = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
            ratio_numeric = numeric_attempt.notnull().mean()
            if ratio_numeric > 0.7:  # umbral configurable
                df[col] = numeric_attempt
                conversion_log.append(f"🔢 La columna **'{col}'** fue convertida a **numérica**.")
            else:
                df[col] = df[col].astype("category")
                conversion_log.append(f"🏷️ La columna **'{col}'** fue clasificada como **categórica**.")

    # --- Clasificación final de variables ---
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    st.write("### 🗂️ Clasificación de variables detectadas")
    st.write(f"**Numéricas:** {numeric_cols}")
    st.write(f"**Categóricas:** {cat_cols}")
    st.write(f"**Fechas:** {date_cols}")

    if conversion_log:
        with st.expander("🔎 Ver detalles de conversiones automáticas"):
            for log in conversion_log:
                st.markdown(f"- {log}")

    # --- Valores nulos ---
    st.write("### ❌ Datos Nulos por Columna")
    st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Columna", 0: "Nulos"}))

    # --- Outliers ---
    st.write("### 📉 Resumen de Valores Atípicos")
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_summary[col] = outliers
    st.dataframe(pd.DataFrame.from_dict(outlier_summary, orient="index", columns=["Outliers"]))

    # --- Estadísticas descriptivas ---
    st.write("### 📊 Estadísticas de variables numéricas")
    st.dataframe(df[numeric_cols].describe().T)

    # --- Boxplots (con opción de normalizar) ---
    st.write("### 📦 Boxplots de variables numéricas")
    normalize = st.checkbox("Normalizar datos con MinMaxScaler antes de graficar")
    data_for_boxplot = df[numeric_cols].copy()

    if normalize:
        scaler = MinMaxScaler()
        data_for_boxplot[numeric_cols] = scaler.fit_transform(data_for_boxplot[numeric_cols])
        st.info("Se aplicó MinMaxScaler a los datos numéricos.")

    fig, axes = plt.subplots(nrows=(len(numeric_cols) + 1) // 2, ncols=2, figsize=(12, 6))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=data_for_boxplot[col], ax=axes[i], color="skyblue")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

    # --- Histogramas para categóricas ---
    if cat_cols:
        st.write("### 📊 Histogramas de variables categóricas")
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=df[col], ax=ax, palette="Set2")
            ax.set_title(f"Distribución de {col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # --- Heatmap de correlación ---
    if numeric_cols:
        st.write("### 🔥 Heatmap de correlación")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)

    # --- Correlación entre dos variables ---
    if len(numeric_cols) >= 2:
        st.write("### 🔗 Análisis de correlación entre dos variables")
        col1 = st.selectbox("Selecciona la primera variable", numeric_cols)
        col2 = st.selectbox("Selecciona la segunda variable", numeric_cols)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
        ax.set_title(f"Correlación entre {col1} y {col2}")
        st.pyplot(fig)

    # --- Gráficos de tendencia si hay fechas ---
    if date_cols and numeric_cols:
        st.write("### 📈 Gráficos de tendencia en el tiempo")
        date_col = date_cols[0]  # usar la primera columna fecha encontrada
        trend_var = st.selectbox("Selecciona variable numérica para graficar tendencia", numeric_cols)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=df[date_col], y=df[trend_var], ax=ax)
        ax.set_title(f"Tendencia de {trend_var} en el tiempo ({date_col})")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Pivot Table con stock index ---
    if "stock index" in df.columns and date_cols:
        st.write("### 📊 Pivot Table: Promedio por fecha y Stock Index")
        date_col = date_cols[0]

        pivot_table = pd.pivot_table(
            df,
            values=numeric_cols,
            index=date_col,
            columns="stock index",
            aggfunc="mean"
        )

        st.dataframe(pivot_table)
