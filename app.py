import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==============================
# ⚙️ Configuración general
# ==============================
st.set_page_config(page_title="EDA Interactivo", layout="wide")

# 🎨 Paleta de colores por sección
SECTION_COLORS = {
    "Descripción General": "#F3F9FF",   # Azul muy claro
    "Análisis Numérico": "#FFF5E6",     # Naranja claro
    "Análisis Categórico": "#F0FFF0",   # Verde menta
    "Correlación": "#FFF0F5",           # Rosado claro
    "Tendencias": "#F9FFF3",            # Verde pastel
    "Pivot Table": "#FFFFE0"            # Amarillo suave
}

# ==============================
# 🎨 Sidebar con estilo
# ==============================
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
    
    st.markdown("### 📂 Cargar CSV")
    uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])

# ==============================
# 📂 Carga de datos
# ==============================
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8")
    except Exception:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin-1")

    # Conversión automática de fechas
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='ignore')
        except Exception:
            pass

    # Detectar tipos de columnas
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

    # ==============================
    # 📊 Descripción General
    # ==============================
    if menu == "Descripción General":
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("📊 Descripción General de los Datos")

        st.write("### Primeras filas del dataset")
        st.dataframe(df.head())

        st.write("### Resumen de columnas")
        info_df = pd.DataFrame({
            "Columna": df.columns,
            "Tipo de dato": df.dtypes.astype(str),
            "Valores nulos": df.isnull().sum().values,
            "Valores únicos": [df[col].nunique() for col in df.columns],
            "Ejemplo": [df[col].dropna().iloc[0] if df[col].notnull().any() else None for col in df.columns]
        })
        st.dataframe(info_df)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==============================
    # 🔢 Análisis Numérico
    # ==============================
    elif menu == "Análisis Numérico" and numeric_cols:
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("🔢 Análisis de Variables Numéricas")
        st.write(df[numeric_cols].describe())
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==============================
    # 🔠 Análisis Categórico
    # ==============================
    elif menu == "Análisis Categórico" and categorical_cols:
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("🔠 Análisis de Variables Categóricas")
        for col in categorical_cols:
            st.write(f"#### Distribución de {col}")
            st.bar_chart(df[col].value_counts())
        st.markdown("</div>", unsafe_allow_html=True)

    # ==============================
    # 🔗 Correlación
    # ==============================
    elif menu == "Correlación" and len(numeric_cols) >= 2:
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("🔗 Correlación entre dos variables")

        var1 = st.selectbox("Variable 1", numeric_cols, index=0)
        var2 = st.selectbox("Variable 2", numeric_cols, index=1)

        normalize = st.checkbox("Normalizar datos con MinMaxScaler")

        if var1 == var2:
            st.error("❌ No tiene sentido correlacionar una variable consigo misma. Selecciona dos diferentes.")
        else:
            data_corr = df[[var1, var2]].dropna()
            if normalize:
                scaler = MinMaxScaler()
                data_corr = pd.DataFrame(scaler.fit_transform(data_corr), columns=[var1, var2])

            corr_value = data_corr[var1].corr(data_corr[var2])
            st.success(f"📈 El índice de correlación de Pearson entre **{var1}** y **{var2}** es: `{corr_value:.3f}`")

            fig, ax = plt.subplots()
            sns.scatterplot(data=data_corr, x=var1, y=var2, ax=ax)
            st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==============================
    # 📈 Tendencias
    # ==============================
    elif menu == "Tendencias" and date_cols and numeric_cols:
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("📈 Análisis de Tendencias")

        date_col = st.selectbox("Selecciona la variable de fecha", date_cols)
        num_col = st.selectbox("Selecciona la variable numérica", numeric_cols)
        period = st.selectbox("Periodo de agregación", ["Día", "Mes", "Trimestre", "Año"])

        df_trend = df[[date_col, num_col]].dropna().copy()
        df_trend = df_trend.sort_values(by=date_col)

        if period == "Día":
            df_trend = df_trend.groupby(pd.Grouper(key=date_col, freq="D"))[num_col].mean()
        elif period == "Mes":
            df_trend = df_trend.groupby(pd.Grouper(key=date_col, freq="M"))[num_col].mean()
        elif period == "Trimestre":
            df_trend = df_trend.groupby(pd.Grouper(key=date_col, freq="Q"))[num_col].mean()
        elif period == "Año":
            df_trend = df_trend.groupby(pd.Grouper(key=date_col, freq="Y"))[num_col].mean()

        st.line_chart(df_trend)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==============================
    # 📊 Pivot Table
    # ==============================
    elif menu == "Pivot Table" and categorical_cols and numeric_cols:
        st.markdown(f"<div style='background-color:{SECTION_COLORS[menu]}; padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.header("📊 Pivot Table")
        cat1 = st.selectbox("Variable categórica (filas)", categorical_cols, index=0)
        cat2 = st.selectbox("Variable categórica (columnas)", categorical_cols, index=1 if len(categorical_cols) > 1 else 0)
        num = st.selectbox("Variable numérica (valores)", numeric_cols, index=0)

        pivot = pd.pivot_table(df, values=num, index=cat1, columns=cat2, aggfunc="mean", fill_value=0)
        st.dataframe(pivot)
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# 👇 Créditos al final
# ==============================
st.sidebar.markdown("---")
st.sidebar.caption("✨ App creada con Streamlit")
