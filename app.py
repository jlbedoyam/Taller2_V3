import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================
st.set_page_config(layout="wide", page_title="EDA Interactivo")

# CSS para estilos
st.markdown("""
<style>
    .main {
        background-color: #f4f6f9;
        padding: 20px;
        border-radius: 15px;
    }
    .sidebar .sidebar-content {
        background-color: #a8c9ff;
        color: black;
        padding: 20px;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("📊 Menú de navegación")
menu = st.sidebar.radio("Ir a sección:", [
    "Carga de datos",
    "Descripción general",
    "EDA numérico y categórico",
    "Correlaciones",
    "Tendencias",
    "Pivot Table"
])

st.sidebar.markdown("---")
st.sidebar.markdown("✨ Hecho con Streamlit")

# ==============================
# VARIABLES GLOBALES
# ==============================
if "df" not in st.session_state:
    st.session_state.df = None

# ==============================
# SECCIÓN: CARGA DE DATOS
# ==============================
if menu == "Carga de datos":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("📂 Carga de Datos")
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Identificar fechas
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > 0.9 * len(parsed):  # si la mayoría son fechas válidas
                    df[col] = parsed
            except Exception:
                pass

        # Convertir a numéricos donde se pueda
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(",", "").str.replace("$", ""), errors="coerce")
                except Exception:
                    pass

        st.session_state.df = df
        st.success("✅ Datos cargados correctamente")

        st.subheader("Vista previa")
        st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# SECCIÓN: DESCRIPCIÓN GENERAL
# ==============================
elif menu == "Descripción general":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<div class='main' style='background-color:#eaf2f8;'>", unsafe_allow_html=True)
        st.title("📋 Descripción General de los Datos")

        st.write("### Información de columnas")
        buffer = []
        df.info(buf=buffer)
        st.text("".join(buffer))

        st.write("### Tipos de datos detectados:")
        st.write(df.dtypes)

        st.write("### Resumen estadístico")
        st.write(df.describe(include="all"))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Primero carga un dataset en la sección *Carga de datos*.")

# ==============================
# SECCIÓN: EDA NUMÉRICO Y CATEGÓRICO
# ==============================
elif menu == "EDA numérico y categórico":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<div class='main' style='background-color:#fef9e7;'>", unsafe_allow_html=True)
        st.title("🔎 Exploración de Datos")

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        st.subheader("Valores nulos")
        st.write(df.isnull().sum())

        if num_cols:
            st.subheader("Medidas de tendencia central")
            st.write(df[num_cols].describe().T)

            normalize = st.checkbox("Normalizar variables numéricas con MinMaxScaler")
            plot_data = df[num_cols].dropna()
            if normalize:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(plot_data)
                plot_data = pd.DataFrame(scaled, columns=num_cols)

            st.subheader("Boxplots")
            fig, axs = plt.subplots(len(num_cols) // 2 + 1, 2, figsize=(10, 5 * (len(num_cols)//2 + 1)))
            axs = axs.flatten()
            for i, col in enumerate(num_cols):
                sns.boxplot(y=plot_data[col], ax=axs[i])
                axs[i].set_title(col)
            plt.tight_layout()
            st.pyplot(fig)

        if cat_cols:
            st.subheader("Histogramas categóricos")
            for col in cat_cols:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.countplot(x=df[col], ax=ax)
                ax.set_title(col)
                st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Primero carga un dataset en la sección *Carga de datos*.")

# ==============================
# SECCIÓN: CORRELACIONES
# ==============================
elif menu == "Correlaciones":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<div class='main' style='background-color:#fdebd0;'>", unsafe_allow_html=True)
        st.title("📈 Análisis de Correlaciones")

        num_cols = df.select_dtypes(include="number").columns.tolist()

        if num_cols:
            st.subheader("Mapa de calor de correlaciones")
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, cmap="RdYlGn", center=0, annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

            st.subheader("Correlación entre dos variables")
            var1 = st.selectbox("Variable 1", num_cols, index=0)
            var2 = st.selectbox("Variable 2", num_cols, index=1)

            if var1 == var2:
                st.error("❌ No tiene sentido calcular correlación con la misma variable.")
            else:
                corr_value = df[var1].corr(df[var2])
                st.success(f"📌 Coeficiente de Pearson entre **{var1}** y **{var2}**: `{corr_value:.3f}`")

                fig, ax = plt.subplots()
                sns.scatterplot(x=df[var1], y=df[var2], ax=ax)
                ax.set_title(f"Relación entre {var1} y {var2}")
                st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Primero carga un dataset en la sección *Carga de datos*.")

# ==============================
# SECCIÓN: TENDENCIAS
# ==============================
elif menu == "Tendencias":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<div class='main' style='background-color:#e8f8f5;'>", unsafe_allow_html=True)
        st.title("📉 Análisis de Tendencias")

        date_cols = df.select_dtypes(include="datetime").columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if date_cols and num_cols:
            date_col = st.selectbox("Selecciona variable de fecha", date_cols)
            trend_var = st.selectbox("Selecciona variable numérica", num_cols)
            period = st.selectbox("Periodo de agrupación", ["D", "W", "M", "Q", "Y"], index=2)

            trend_data = df.groupby(pd.Grouper(key=date_col, freq=period))[trend_var].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10,5))
            sns.lineplot(x=date_col, y=trend_var, data=trend_data, ax=ax)
            ax.set_title(f"Tendencia de {trend_var} agrupado por {period}")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No se detectaron variables de fecha y numéricas.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Primero carga un dataset en la sección *Carga de datos*.")

# ==============================
# SECCIÓN: PIVOT TABLE
# ==============================
elif menu == "Pivot Table":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<div class='main' style='background-color:#f9ebea;'>", unsafe_allow_html=True)
        st.title("📊 Pivot Table")

        date_cols = df.select_dtypes(include="datetime").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if date_cols and cat_cols and num_cols:
            date_col = st.selectbox("Selecciona fecha", date_cols)
            cat_col = st.selectbox("Selecciona columna categórica (ej. Stock Index)", cat_cols)
            num_col = st.selectbox("Selecciona variable numérica", num_cols)

            pivot = pd.pivot_table(df, index=date_col, columns=cat_col, values=num_col, aggfunc="mean")

            st.write("### Tabla pivote")
            st.dataframe(pivot)

            st.write("### Gráfico de tendencias por categoría")
            fig, ax = plt.subplots(figsize=(10,5))
            pivot.plot(ax=ax)
            ax.set_title(f"{num_col} promedio por {cat_col} en el tiempo")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Se requieren al menos una fecha, una categórica y una numérica.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Primero carga un dataset en la sección *Carga de datos*.")
