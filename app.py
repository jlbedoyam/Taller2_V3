import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

# Configuración general
st.set_page_config(page_title="EDA Interactivo", layout="wide")

# Paleta de colores amigable
SECTION_COLORS = {
    "Descripción General": "#F3F9FF",   # Azul muy claro
    "Análisis Numérico": "#FFF5E6",     # Naranja claro
    "Análisis Categórico": "#F0FFF0",   # Verde menta
    "Correlación": "#FFF0F5",           # Rosado claro
    "Tendencias": "#F9FFF3",            # Verde pastel
    "Pivot Table": "#FFFFE0"            # Amarillo suave
}

# Sidebar con estilo
st.markdown("""
    <style>
    [data-testid=stSidebar] {
        background-color: #87CEFA; /* azul claro */
        color: black;
    }
    .sidebar-title {
        font-size: 22px !important;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='sidebar-title'>📊 Menú Principal</div>", unsafe_allow_html=True)
    menu = st.radio("Ir a la sección:", 
                    ["Descripción General", "Análisis Numérico", "Análisis Categórico", 
                     "Correlación", "Tendencias", "Pivot Table"])

# Subir archivo
st.sidebar.markdown("### 📂 Cargar CSV")
uploaded_file = st.sidebar.file_uploader("Elige un archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Conversión de fechas
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except Exception:
            pass

    # Detectar tipos
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # ===== Sección: Descripción General =====
    if menu == "Descripción General":
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("📊 Descripción General de los Datos")
        st.write(df.head())
        st.write("**Información del dataset:**")
        st.write(df.info())
        st.write("**Datos nulos por columna:**")
        st.write(df.isnull().sum())
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== Sección: Análisis Numérico =====
    elif menu == "Análisis Numérico":
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("📈 Análisis Numérico")
        st.write(df[numeric_cols].describe())

        # Outliers
        st.subheader("🔎 Detección de valores atípicos (Z-score > 3)")
        outliers = (df[numeric_cols].apply(zscore).abs() > 3).sum()
        st.write(outliers)

        normalize = st.checkbox("Normalizar con MinMaxScaler antes de boxplots", key="normalize_boxplot")
        data_plot = df[numeric_cols]
        if normalize:
            scaler = MinMaxScaler()
            data_plot = pd.DataFrame(scaler.fit_transform(data_plot), columns=numeric_cols)

        fig, axes = plt.subplots(nrows=(len(numeric_cols) // 2) + 1, ncols=2, figsize=(12, 6))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=data_plot[col], ax=axes[i])
            axes[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== Sección: Análisis Categórico =====
    elif menu == "Análisis Categórico":
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("📊 Análisis Categórico")
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(col)
            st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== Sección: Correlación =====
    elif menu == "Correlación":
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("🔗 Análisis de Correlación")
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="RdYlGn", ax=ax, center=0, vmin=-1, vmax=1)
        st.pyplot(fig)

        normalize_corr = st.checkbox("Normalizar variables antes de correlación", key="normalize_corr")
        data_corr = df[numeric_cols]
        if normalize_corr:
            scaler = MinMaxScaler()
            data_corr = pd.DataFrame(scaler.fit_transform(data_corr), columns=numeric_cols)

        var1 = st.selectbox("Variable 1", numeric_cols, index=0)
        var2 = st.selectbox("Variable 2", numeric_cols, index=1)
        if var1 == var2:
            st.error("⚠️ No tiene sentido correlacionar una variable consigo misma.")
        else:
            corr_value = data_corr[var1].corr(data_corr[var2])
            st.write(f"Coeficiente de correlación de Pearson entre **{var1}** y **{var2}**: **{corr_value:.2f}**")
            fig, ax = plt.subplots()
            ax.scatter(data_corr[var1], data_corr[var2])
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== Sección: Tendencias =====
    elif menu == "Tendencias":
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("📅 Análisis de Tendencias")
        if date_cols:
            date_col = date_cols[0]
            trend_var = st.selectbox("Selecciona variable numérica", numeric_cols)
            period = st.selectbox("Selecciona periodo de resumen", ["D", "W", "M", "Q", "Y"], index=2)

            trend_data = df.groupby(pd.Grouper(key=date_col, freq=period))[trend_var].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trend_data[date_col], trend_data[trend_var], marker='o')
            ax.set_title(f"Tendencia de {trend_var} ({period})")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No se detectaron columnas de tipo fecha.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== Sección: Pivot Table =====
    elif menu == "Pivot Table":
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("📊 Pivot Table con Stock Index")
        if "stock index" in df.columns and date_cols:
            stock_col = "stock index"
            date_col = date_cols[0]
            pivot = pd.pivot_table(df, values=numeric_cols, index=date_col, columns=stock_col, aggfunc='mean')
            st.dataframe(pivot)
        else:
            st.warning("⚠️ No se encontró columna 'stock index' o columna de tipo fecha.")
        st.markdown("</div>", unsafe_allow_html=True)

# Créditos al final
st.markdown("<hr><center><small>✨ Aplicación desarrollada con Streamlit ✨</small></center>", unsafe_allow_html=True)
