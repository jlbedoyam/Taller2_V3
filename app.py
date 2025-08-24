import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# ======================
# ESTILOS
# ======================
st.set_page_config(layout="wide", page_title="EDA Interactivo")

st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #e6f0ff;
    }
    .reportview-container {
        background: #fdfdfd;
    }
    .main {
        background-color: #fafafa;
    }
    h1, h2, h3 {
        color: #003366;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio(
    "📊 Menú de navegación",
    ["Carga de datos", "Descripción general", "Análisis de valores nulos y atípicos",
     "Visualización numérica", "Visualización categórica", "Correlaciones",
     "Análisis de tendencias", "Pivot Table"]
)

# ======================
# CARGA DE DATOS
# ======================
if "df" not in st.session_state:
    st.session_state.df = None

st.sidebar.markdown("---")
st.sidebar.info("App desarrollada con ❤️ usando Streamlit")

if menu == "Carga de datos":
    st.markdown("<div style='background-color:#f0f8ff;padding:20px;border-radius:10px'>", unsafe_allow_html=True)
    st.header("📂 Carga de datos")
    file = st.file_uploader("Sube tu archivo CSV", type="csv")
    if file:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1")

        # Detectar columnas de fecha
        for col in df.columns:
            try:
                if pd.to_datetime(df[col], errors="coerce").notnull().sum() > 0.8 * len(df):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                continue

        st.session_state.df = df
        st.success("✅ Datos cargados correctamente")
        st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# DESCRIPCIÓN GENERAL
# ======================
if menu == "Descripción general" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📖 Descripción general")
    st.write("### Tipos de datos detectados")
    st.write(df.dtypes)

    st.write("### Resumen estadístico (numéricas)")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 0:
        st.write(df[num_cols].describe())
    else:
        st.warning("No se detectaron variables numéricas.")

# ======================
# NULOS Y ATÍPICOS
# ======================
if menu == "Análisis de valores nulos y atípicos" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("🔎 Valores nulos y atípicos")

    st.subheader("Valores nulos por columna")
    st.write(df.isnull().sum())

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 0:
        st.subheader("Valores atípicos (Z-score > 3)")
        outliers = (df[num_cols].apply(zscore).abs() > 3).sum()
        st.write(outliers)

# ======================
# VISUALIZACIÓN NUMÉRICA
# ======================
if menu == "Visualización numérica" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Boxplots de variables numéricas")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 0:
        normalize = st.checkbox("Normalizar datos con MinMaxScaler", value=False)
        data_plot = df[num_cols].copy()
        if normalize:
            scaler = MinMaxScaler()
            data_plot = pd.DataFrame(scaler.fit_transform(data_plot), columns=num_cols)

        fig, axes = plt.subplots(nrows=(len(num_cols) // 2) + 1, ncols=2, figsize=(12, 6))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.boxplot(y=data_plot[col], ax=axes[i], color="skyblue")
            axes[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)

# ======================
# VISUALIZACIÓN CATEGÓRICA
# ======================
if menu == "Visualización categórica" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Histogramas de variables categóricas")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind="bar", ax=ax, color="lightcoral")
            ax.set_title(f"Frecuencia de {col}")
            st.pyplot(fig)

# ======================
# CORRELACIONES
# ======================
if menu == "Correlaciones" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Correlaciones entre variables")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 1:
        # Heatmap
        st.subheader("Matriz de correlación")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)

        # Selección de variables
        st.subheader("Correlación entre dos variables")
        normalize_corr = st.checkbox("Normalizar antes de correlacionar", value=False)

        col1, col2 = st.columns(2)
        var1 = col1.selectbox("Variable 1", num_cols, index=0)
        var2 = col2.selectbox("Variable 2", num_cols, index=1 if len(num_cols) > 1 else 0)

        if var1 == var2:
            st.error("❌ No tiene sentido correlacionar la misma variable.")
        else:
            data_corr = df[[var1, var2]].dropna()
            if normalize_corr:
                scaler = MinMaxScaler()
                data_corr = pd.DataFrame(scaler.fit_transform(data_corr), columns=[var1, var2])
            corr_value = data_corr[var1].corr(data_corr[var2])
            st.info(f"Coeficiente de correlación de Pearson: **{corr_value:.4f}**")

# ======================
# ANÁLISIS DE TENDENCIAS
# ======================
if menu == "Análisis de tendencias" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📈 Análisis de tendencias")

    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(date_cols) > 0 and len(num_cols) > 0:
        date_col = st.selectbox("Selecciona columna de fecha", date_cols)
        trend_var = st.selectbox("Selecciona variable numérica", num_cols)

        period = st.radio("Periodo de resumen", ["Día", "Mes", "Trimestre", "Año"], horizontal=True)
        freq_map = {"Día": "D", "Mes": "M", "Trimestre": "Q", "Año": "Y"}
        freq = freq_map[period]

        trend_data = df.groupby(pd.Grouper(key=date_col, freq=freq))[trend_var].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=trend_data, x=date_col, y=trend_var, ax=ax, marker="o")
        ax.set_title(f"Tendencia de {trend_var} por {period}")
        st.pyplot(fig)
    else:
        st.warning("No hay columna de fecha y variable numérica disponible.")

# ======================
# PIVOT TABLE
# ======================
if menu == "Pivot Table" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Pivot Table con promedio")

    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(date_cols) > 0 and len(cat_cols) > 0 and len(num_cols) > 0:
        date_col = st.selectbox("Selecciona columna de fecha", date_cols)
        cat_col = st.selectbox("Selecciona columna categórica (ej. Stock Index)", cat_cols)
        num_var = st.selectbox("Selecciona variable numérica", num_cols)

        pivot = pd.pivot_table(df, index=date_col, columns=cat_col, values=num_var, aggfunc="mean")

        st.dataframe(pivot.head())
    else:
        st.warning("Se necesitan al menos una columna de fecha, una categórica y una numérica.")
