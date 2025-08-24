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
        background: linear-gradient(180deg, #66b2ff, #004080);
        color: white;
    }
    .mainframe {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: black;
    }
    .correlation {background: linear-gradient(135deg, #ffecd2, #fcb69f);}
    .eda {background: linear-gradient(135deg, #fff6a3, #ffd966);}
    .trend {background: linear-gradient(135deg, #a8edea, #fed6e3);}
    .pivot {background: linear-gradient(135deg, #d4a5ff, #fbc2eb);}
    .credits {text-align:center; font-size: 13px; color: gray; margin-top: 60px;}
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
    with st.expander("📂 **Carga de datos**", expanded=True):
        st.markdown('<div class="mainframe eda">', unsafe_allow_html=True)

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
            st.dataframe(df.head().style.background_gradient(cmap="coolwarm"))

        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# EDA
# =============================
if menu == "EDA" and st.session_state.df is not None:
    df = st.session_state.df
    with st.expander("🔍 **Análisis Exploratorio de Datos (EDA)**", expanded=True):
        st.markdown('<div class="mainframe eda">', unsafe_allow_html=True)

        st.subheader("📊 Resumen estadístico")
        st.dataframe(df.describe(include="all").style.background_gradient(cmap="plasma"))

        st.subheader("❌ Datos nulos por columna")
        st.dataframe(df.isnull().sum().to_frame("Nulos").style.background_gradient(cmap="YlOrRd"))

        st.subheader("📦 Valores atípicos (IQR)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_summary = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            outliers_summary[col] = len(outliers)
        st.dataframe(pd.DataFrame.from_dict(outliers_summary, orient="index", columns=["Cantidad de atípicos"]).style.background_gradient(cmap="coolwarm"))

        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# ANÁLISIS DE CORRELACIÓN
# =============================
if menu == "Análisis de correlación" and st.session_state.df is not None:
    df = st.session_state.df
    with st.expander("📈 **Análisis de Correlación**", expanded=True):
        st.markdown('<div class="mainframe correlation">', unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            normalize_corr = st.checkbox("🔄 Normalizar datos (MinMaxScaler)")

            if normalize_corr:
                scaler = MinMaxScaler()
                data_corr = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
            else:
                data_corr = df[numeric_cols]

            st.subheader("🔥 Heatmap de correlaciones")
            corr_matrix = data_corr.corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr_matrix, annot=True, cmap="Spectral", center=0, ax=ax)
            st.pyplot(fig)

            var1 = st.selectbox("Selecciona la primera variable:", numeric_cols, index=0)
            var2 = st.selectbox("Selecciona la segunda variable:", numeric_cols, index=1)

            if var1 == var2:
                st.error("⚠️ No tiene sentido correlacionar una variable consigo misma.")
            else:
                corr_value = data_corr[var1].corr(data_corr[var2])
                st.success(f"📌 Correlación de Pearson entre **{var1}** y **{var2}**: **{corr_value:.2f}**")

                fig, ax = plt.subplots()
                sns.scatterplot(x=data_corr[var1], y=data_corr[var2], hue=data_corr[var1], palette="coolwarm", ax=ax)
                st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# ANÁLISIS DE TENDENCIA
# =============================
if menu == "Análisis de tendencia" and st.session_state.df is not None:
    df = st.session_state.df
    with st.expander("📅 **Análisis de Tendencia Temporal**", expanded=True):
        st.markdown('<div class="mainframe trend">', unsafe_allow_html=True)

        date_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        if len(date_cols) > 0:
            date_col = st.selectbox("Selecciona la variable de tipo fecha:", date_cols)
            num_trend_vars = df.select_dtypes(include=[np.number]).columns
            trend_var = st.selectbox("Selecciona la variable numérica:", num_trend_vars)

            period_dict = {"D": "Diario", "W": "Semanal", "M": "Mensual", "Q": "Trimestral", "Y": "Anual"}
            period = st.selectbox("Selecciona el periodo:", list(period_dict.keys()), format_func=lambda x: period_dict[x])

            trend_data = df.groupby(pd.Grouper(key=date_col, freq=period))[trend_var].mean().reset_index()

            fig, ax = plt.subplots()
            sns.lineplot(data=trend_data, x=date_col, y=trend_var, marker="o", linewidth=2.5, color="#ff5733")
            st.pyplot(fig)
        else:
            st.info("⚠️ No se detectaron columnas de tipo fecha.")

        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PIVOT TABLE
# =============================
if menu == "Pivot Table" and st.session_state.df is not None:
    df = st.session_state.df
    with st.expander("📊 **Pivot Table con Stock Index**", expanded=True):
        st.markdown('<div class="mainframe pivot">', unsafe_allow_html=True)

        if "stock index" in df.columns:
            pivot_table = pd.pivot_table(
                df,
                values=df.select_dtypes(include=[np.number]).columns.tolist(),
                index="date" if "date" in df.columns else df.index,
                columns="stock index",
                aggfunc=np.mean
            )
            st.dataframe(pivot_table.style.background_gradient(cmap="coolwarm"))
        else:
            st.error("⚠️ No existe la columna `stock index` en el dataset.")

        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# CREDITOS
# =============================
st.markdown('<div class="credits">🌟 Creado con ❤️ y muchos colores usando Streamlit</div>', unsafe_allow_html=True)
