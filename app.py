import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configuración general de la app
st.set_page_config(page_title="EDA Automático", layout="wide")

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Exploratory Data Analysis (EDA) Automático")

# Cargar archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Detectar columnas por tipo
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    # Intentar parsear fechas si hay columnas tipo fecha
    if not date_cols:
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
                date_cols.append(col)
            except:
                pass

    # --- Descripción inicial ---
    st.write("### 📝 Descripción general de los datos")
    st.write("**Dimensiones del dataset:**", df.shape)
    st.write("**Tipos de datos:**")
    st.write(df.dtypes)
    st.dataframe(df.head())

    # --- Datos nulos y atípicos ---
    st.write("### ❓ Resumen de valores nulos y atípicos")
    nulls = df.isnull().sum()
    outliers = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers[col] = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
    summary_nulls = pd.DataFrame({"Nulos": nulls, "Atípicos": pd.Series(outliers)})
    st.dataframe(summary_nulls)

    # --- Estadísticas de variables numéricas ---
    st.write("### 📈 Estadísticas de variables numéricas")
    st.write(df[numeric_cols].describe())

    # --- Boxplots ---
    st.write("### 📦 Boxplots de variables numéricas")
    normalize = st.checkbox("Normalizar variables con MinMaxScaler (0-1)", key="boxplot_norm")
    box_data = df[numeric_cols].dropna()
    if normalize:
        scaler = MinMaxScaler()
        box_data = pd.DataFrame(scaler.fit_transform(box_data), columns=numeric_cols)

    fig, axes = plt.subplots(
        nrows=(len(numeric_cols) // 2) + 1, ncols=2, figsize=(12, len(numeric_cols) * 2.5)
    )
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=box_data[col], ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    st.pyplot(fig)

    # --- Histogramas de categóricas ---
    st.write("### 📊 Histogramas de variables categóricas")
    for col in categorical_cols:
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        ax.set_title(f"Frecuencia de {col}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Heatmap de correlación ---
    if len(numeric_cols) > 1:
        st.write("### 🌡️ Mapa de calor de correlaciones")
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = sns.diverging_palette(10, 240, as_cmap=True)  # Verde → Amarillo → Rojo
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0, ax=ax)
        st.pyplot(fig)

    # --- Análisis de correlación entre dos variables ---
    st.write("### 🔗 Análisis de correlación entre dos variables numéricas")

    if len(numeric_cols) >= 2:
        var1 = st.selectbox("Selecciona la primera variable", numeric_cols)
        var2 = st.selectbox("Selecciona la segunda variable", numeric_cols)

        # Checkbox para normalización con MinMaxScaler
        normalize_corr = st.checkbox("Normalizar variables con MinMaxScaler (0-1)", key="corr_norm")

        data_corr = df[[var1, var2]].dropna()

        if normalize_corr:
            scaler = MinMaxScaler()
            data_corr[[var1, var2]] = scaler.fit_transform(data_corr[[var1, var2]])

        # Forzamos a Series para evitar error de ambigüedad
        x = data_corr[var1].astype(float).squeeze()
        y = data_corr[var2].astype(float).squeeze()

        corr_value = x.corr(y)

        st.write(f"Coeficiente de correlación entre **{var1}** y **{var2}**: `{corr_value:.2f}`")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=x, y=y, ax=ax)
        ax.set_title(f"Correlación {var1} vs {var2}")
        st.pyplot(fig)

    # --- Análisis de tendencias ---
    if date_cols and numeric_cols:
        st.write("### 📅 Análisis de tendencias")
        date_col = st.selectbox("Selecciona la variable de fecha", date_cols)
        trend_var = st.selectbox("Selecciona la variable numérica a graficar", numeric_cols)
        period = st.selectbox("Periodo de resumen", ["Día", "Semana", "Mes", "Trimestre", "Año"])

        freq_map = {
            "Día": "D",
            "Semana": "W",
            "Mes": "M",
            "Trimestre": "Q",
            "Año": "Y"
        }

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        trend_data = (
            df.groupby(pd.Grouper(key=date_col, freq=freq_map[period]))[trend_var]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=trend_data, x=date_col, y=trend_var, ax=ax)
        ax.set_title(f"Tendencia de {trend_var} ({period})")
        st.pyplot(fig)

    # --- Pivot table con Stock Index ---
    if "Stock Index" in df.columns and date_cols:
        st.write("### 📊 Pivot Table con Stock Index")
        date_col = date_cols[0]
        pivot_table = pd.pivot_table(
            df,
            values=numeric_cols,
            index=date_col,
            columns="Stock Index",
            aggfunc="mean"
        )
        st.dataframe(pivot_table)
