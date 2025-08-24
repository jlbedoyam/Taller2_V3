import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Configuración general de la página
st.set_page_config(page_title="EDA Automático", layout="wide")

# --- CSS para mejorar presentación ---
st.markdown(
    """
    <style>
    .main {background-color: #F9F9F9;}
    h1, h2, h3 {color: #2C3E50;}
    .stDataFrame {background-color: white; border-radius: 10px; padding: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Explorador Automático de Datos (EDA)")

# --- Carga de dataset ---
uploaded_file = st.file_uploader("Carga un archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("👀 Vista previa de los datos")
    st.dataframe(df.head())

    # Identificación de tipos de variables
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Forzar conversión a datetime si alguna columna parece ser fecha
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                pass

    st.write("### 🧾 Información del dataset")
    st.write(f"**Columnas numéricas:** {numeric_cols}")
    st.write(f"**Columnas categóricas:** {cat_cols}")
    st.write(f"**Columnas de fecha:** {date_cols}")

    # --- Datos nulos y outliers ---
    st.write("### ⚠️ Datos nulos y valores atípicos")
    nulls = df.isnull().sum()
    st.write("**Valores nulos por columna:**")
    st.dataframe(nulls)

    if numeric_cols:
        z_scores = stats.zscore(df[numeric_cols].dropna())
        outliers = (abs(z_scores) > 3).sum(axis=0)
        st.write("**Valores atípicos detectados (z-score > 3):**")
        st.dataframe(pd.Series(outliers, index=numeric_cols))

    # --- Estadísticas descriptivas ---
    st.write("### 📈 Estadísticas descriptivas (variables numéricas)")
    st.write(df[numeric_cols].describe())

    # --- Boxplots con opción de normalización ---
    if numeric_cols:
        st.write("### 📦 Distribución (Boxplots)")
        normalize = st.checkbox("Normalizar variables con MinMaxScaler", key="boxplot_norm")
        data_plot = df[numeric_cols].copy()

        if normalize:
            scaler = MinMaxScaler()
            data_plot[numeric_cols] = scaler.fit_transform(data_plot[numeric_cols])

        fig, axes = plt.subplots(nrows=len(numeric_cols)//2 + len(numeric_cols)%2, ncols=2, figsize=(12, 4*len(numeric_cols)//2))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=data_plot, y=col, ax=axes[i])
        plt.tight_layout()
        st.pyplot(fig)

    # --- Histogramas categóricos ---
    if cat_cols:
        st.write("### 📊 Frecuencia de variables categóricas")
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Frecuencia de {col}")
            st.pyplot(fig)

    # --- Heatmap de correlaciones ---
    if len(numeric_cols) > 1:
        st.write("### 🌡️ Matriz de correlación")
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = sns.diverging_palette(150, 10, as_cmap=True)  # Verde -> Amarillo -> Rojo
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0, ax=ax)
        st.pyplot(fig)

    # --- Análisis de correlación entre dos variables ---
    st.write("### 🔗 Análisis de correlación entre dos variables numéricas")

    if len(numeric_cols) >= 2:
        var1 = st.selectbox("Selecciona la primera variable", numeric_cols)
        var2 = st.selectbox("Selecciona la segunda variable", numeric_cols)

        # Checkbox para normalización con MinMaxScaler
        normalize_corr = st.checkbox("Normalizar variables con MinMaxScaler (0-1)", key="corr_norm")

        x = df[var1].dropna()
        y = df[var2].dropna()
        data_corr = pd.concat([x, y], axis=1).dropna()

        if normalize_corr:
            scaler = MinMaxScaler()
            data_corr[[var1, var2]] = scaler.fit_transform(data_corr[[var1, var2]])

        corr_value = data_corr[var1].corr(data_corr[var2])

        st.write(f"Coeficiente de correlación entre **{var1}** y **{var2}**: `{corr_value:.2f}`")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=data_corr[var1], y=data_corr[var2], ax=ax)
        ax.set_title(f"Correlación {var1} vs {var2}")
        st.pyplot(fig)

    # --- Gráficos de tendencia si hay fecha ---
    if date_cols and numeric_cols:
        st.write("### 📉 Análisis de tendencia en el tiempo")

        date_col = st.selectbox("Selecciona la columna de fecha", date_cols)
        trend_var = st.selectbox("Selecciona la variable numérica a analizar", numeric_cols)

        # Opciones de frecuencia
        period_dict = {"Diario": "D", "Semanal": "W", "Mensual": "M", "Trimestral": "Q", "Anual": "Y"}
        period_label = st.selectbox("Selecciona el periodo de resumen", list(period_dict.keys()))
        period = period_dict[period_label]

        df[date_col] = pd.to_datetime(df[date_col])
        trend_data = df.groupby(pd.Grouper(key=date_col, freq=period))[trend_var].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x=trend_data[date_col], y=trend_data[trend_var], ax=ax)
        ax.set_title(f"Tendencia de {trend_var} ({period_label})")
        st.pyplot(fig)

    # --- Pivot Table con Stock Index ---
    if "stock index" in df.columns and date_cols:
        st.write("### 📊 Pivot Table (Stock Index vs Fecha)")

        date_col = st.selectbox("Selecciona la columna de fecha para Pivot Table", date_cols, key="pivot_date")
        pivot_table = pd.pivot_table(
            df,
            values=numeric_cols,
            index=date_col,
            columns="stock index",
            aggfunc="mean"
        )
        st.dataframe(pivot_table)
