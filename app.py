import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Estilos CSS ---
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        h1, h2, h3 { color: #2c3e50; }
        .stDataFrame { background-color: white; border-radius: 10px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- Título ---
st.title("📊 Aplicación de EDA interactivo")

# --- Carga de archivo ---
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 👀 Vista previa de los datos")
    st.dataframe(df.head())

    # --- Conversión de tipos ---
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

            # 2. Intentar convertir a numérico si mayoría son números
            numeric_attempt = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
            )
            ratio_numeric = numeric_attempt.notnull().mean()
            if ratio_numeric > 0.7:
                df[col] = numeric_attempt
                conversion_log.append(f"🔢 La columna **'{col}'** fue convertida a **numérica**.")
            else:
                df[col] = df[col].astype("category")
                conversion_log.append(f"🏷️ La columna **'{col}'** fue clasificada como **categórica**.")

    st.write("### 🔄 Conversión automática de tipos")
    for log in conversion_log:
        st.markdown(log)

    # --- Identificación de tipos ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="category").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    st.write("### 📑 Información de columnas")
    st.write(f"**Numéricas:** {numeric_cols}")
    st.write(f"**Categóricas:** {cat_cols}")
    st.write(f"**Fechas:** {date_cols}")

    # --- Datos nulos ---
    st.write("### 🧹 Valores nulos por columna")
    st.dataframe(df.isnull().sum())

    # --- Outliers ---
    st.write("### ⚠️ Resumen de valores atípicos")
    outlier_summary = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary[col] = outliers
    st.dataframe(pd.DataFrame.from_dict(outlier_summary, orient="index", columns=["# Outliers"]))

    # --- Estadísticas numéricas ---
    st.write("### 📊 Estadísticas de variables numéricas")
    st.write(df[numeric_cols].describe())

    # --- Normalización opcional ---
    normalize = st.checkbox("Normalizar datos numéricos con MinMaxScaler antes de graficar boxplots")

    if normalize and numeric_cols:
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        box_data = df_scaled
    else:
        box_data = df

    # --- Boxplots agrupados ---
    if numeric_cols:
        st.write("### 📦 Boxplots de variables numéricas")
        fig, axes = plt.subplots(nrows=(len(numeric_cols) + 2) // 3, ncols=3, figsize=(12, 4 * ((len(numeric_cols) + 2) // 3)))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=box_data, y=col, ax=axes[i])
            axes[i].set_title(f"Boxplot de {col}")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)

    # --- Histogramas categóricos ---
    if cat_cols:
        st.write("### 📊 Histogramas de variables categóricas")
        for col in cat_cols:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Distribución de {col}")
            st.pyplot(fig)

    # --- Heatmap de correlación ---
    if numeric_cols:
        st.write("### 🔥 Mapa de calor de correlación")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)

        # --- Correlación entre dos variables seleccionadas ---
        st.write("### 📈 Análisis de correlación entre dos variables")
        col1 = st.selectbox("Selecciona la primera variable", numeric_cols)
        col2 = st.selectbox("Selecciona la segunda variable", numeric_cols)

        if col1 and col2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
            ax.set_title(f"Correlación entre {col1} y {col2}")
            st.pyplot(fig)
            st.write(f"Coeficiente de correlación: {df[col1].corr(df[col2]):.2f}")

    # --- Gráficos de tendencia con periodo ---
    if date_cols and numeric_cols:
        st.write("### 📈 Gráficos de tendencia en el tiempo")
        date_col = date_cols[0]  # usar la primera columna fecha encontrada
        trend_var = st.selectbox("Selecciona variable numérica para graficar tendencia", numeric_cols)

        # Selección de periodo de resumen
        period = st.selectbox(
            "Selecciona el periodo de resumen",
            {"Diario": "D", "Semanal": "W", "Mensual": "M", "Trimestral": "Q", "Anual": "Y"}
        )

        # Agrupación por periodo
        trend_data = df.groupby(pd.Grouper(key=date_col, freq=period))[trend_var].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=trend_data[date_col], y=trend_data[trend_var], ax=ax, marker="o")
        ax.set_title(f"Tendencia de {trend_var} ({period})")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Pivot Table con Stock Index ---
    if "stock index" in df.columns and date_cols and numeric_cols:
        st.write("### 📊 Pivot Table: Stock Index vs Date (media)")
        value_var = st.selectbox("Selecciona la variable numérica a promediar", numeric_cols)

        pivot = pd.pivot_table(
            df,
            values=value_var,
            index=date_cols[0],
            columns="stock index",
            aggfunc="mean"
        )

        st.dataframe(pivot.head())

        fig, ax = plt.subplots(figsize=(10, 4))
        pivot.plot(ax=ax)
        ax.set_title(f"Tendencia de {value_var} por Stock Index")
        plt.xticks(rotation=45)
        st.pyplot(fig)
