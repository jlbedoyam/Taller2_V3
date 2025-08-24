import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =============================
# CONFIGURACIÓN DE PÁGINA
# =============================
st.set_page_config(page_title="📊 Análisis Económico y Financiero", layout="wide")

# =============================
# ESTILOS CSS
# =============================
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #cce5ff;
    }
    .mainframe {
        padding: 20px;
        border-radius: 10px;
    }
    .correlation {background-color: #f2f7ff;}
    .eda {background-color: #fff8e5;}
    .trend {background-color: #e8f5e9;}
    .pivot {background-color: #f3e5f5;}
    .credits {text-align:center; font-size: 12px; color: gray; margin-top: 50px;}
</style>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR MENU
# =============================
st.sidebar.title("📌 Menú de navegación")
menu = st.sidebar.radio("Ir a:", [
    "Carga de datos",
    "EDA",
    "Análisis de correlación",
    "Análisis de tendencia",
    "Pivot Table"
])

# =============================
# CARGA DE DATOS
# =============================
if "df" not in st.session_state:
    st.session_state.df = None

if menu == "Carga de datos":
    st.markdown('<div class="mainframe">', unsafe_allow_html=True)
    st.header("📂 Carga de datos")

    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Convertir columnas de fecha si es posible
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except Exception:
                pass

        st.session_state.df = df
        st.success("✅ Datos cargados correctamente")
        st.write(df.head())
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# EDA
# =============================
if menu == "EDA" and st.session_state.df is not None:
    df = st.session_state.df
    st.markdown('<div class="mainframe eda">', unsafe_allow_html=True)
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")

    st.subheader("📊 Resumen estadístico")
    st.write(df.describe(include="all"))

    st.subheader("❌ Datos nulos por columna")
    st.write(df.isnull().sum())

    st.subheader("📦 Detección de valores atípicos (basado en rango IQR)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
        outliers_summary[col] = len(outliers)
    st.write(pd.DataFrame.from_dict(outliers_summary, orient="index", columns=["Cantidad de atípicos"]))

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# ANÁLISIS DE CORRELACIÓN
# =============================
if menu == "Análisis de correlación" and st.session_state.df is not None:
    df = st.session_state.df
    st.markdown('<div class="mainframe correlation">', unsafe_allow_html=True)
    st.header("📈 Análisis de Correlación")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        normalize_corr = st.checkbox("🔄 Normalizar datos (MinMaxScaler)")

        if normalize_corr:
            scaler = MinMaxScaler()
            data_corr = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
        else:
            data_corr = df[numeric_cols]

        # Heatmap
        st.subheader("Mapa de calor de correlaciones")
        corr_matrix = data_corr.corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)

        # Selección de variables
        var1 = st.selectbox("Selecciona la primera variable:", numeric_cols, index=0)
        var2 = st.selectbox("Selecciona la segunda variable:", numeric_cols, index=1)

        if var1 == var2:
            st.error("⚠️ No tiene sentido calcular la correlación de una variable consigo misma.")
        else:
            corr_value = data_corr[var1].corr(data_corr[var2])
            st.success(f"📌 El índice de correlación de Pearson entre **{var1}** y **{var2}** es: **{corr_value:.2f}**")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data_corr[var1], y=data_corr[var2], ax=ax)
            st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# ANÁLISIS DE TENDENCIA
# =============================
if menu == "Análisis de tendencia" and st.session_state.df is not None:
    df = st.session_state.df
    st.markdown('<div class="mainframe trend">', unsafe_allow_html=True)
    st.header("📅 Análisis de Tendencia Temporal")

    date_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    if len(date_cols) > 0:
        date_col = st.selectbox("Selecciona la variable de tipo fecha:", date_cols)
        num_trend_vars = df.select_dtypes(include=[np.number]).columns
        trend_var = st.selectbox("Selecciona la variable numérica:", num_trend_vars)

        period_dict = {
            "D": "Diario",
            "W": "Semanal",
            "M": "Mensual",
            "Q": "Trimestral",
            "Y": "Anual"
        }
        period = st.selectbox("Selecciona el periodo de resumen:", list(period_dict.keys()),
                              format_func=lambda x: period_dict[x])

        trend_data = df.groupby(pd.Grouper(key=date_col, freq=period))[trend_var].mean().reset_index()

        st.line_chart(trend_data.set_index(date_col)[trend_var])
    else:
        st.info("⚠️ No se detectaron variables de tipo fecha.")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PIVOT TABLE
# =============================
if menu == "Pivot Table" and st.session_state.df is not None:
    df = st.session_state.df
    st.markdown('<div class="mainframe pivot">', unsafe_allow_html=True)
    st.header("📊 Pivot Table con Stock Index")

    if "stock index" in df.columns:
        pivot_table = pd.pivot_table(
            df,
            values=df.select_dtypes(include=[np.number]).columns.tolist(),
            index="date" if "date" in df.columns else df.index,
            columns="stock index",
            aggfunc=np.mean
        )
        st.write(pivot_table)
    else:
        st.error("⚠️ No existe la columna `stock index` en el dataset.")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# CREDITOS
# =============================
st.markdown('<div class="credits">Creado con ❤️ usando Streamlit</div>', unsafe_allow_html=True)
