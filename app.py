import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# --- Configuración de página ---
st.set_page_config(
    page_title="Dashboard de Análisis Económico",
    page_icon="📊",
    layout="wide"
)

# --- Variables globales ---
if "df" not in st.session_state:
    st.session_state.df = None

# --- CSS personalizado para sidebar ---
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E90FF, #00BFFF);
        color: white;
        padding: 20px;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6, 
    [data-testid="stSidebar"] p {
        color: white;
    }
    .sidebar-title {
        font-size: 22px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar-footer {
        position: absolute;
        bottom: 20px;
        font-size: 12px;
        text-align: center;
        color: #f0f0f0;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# 📌 Menú en Sidebar
# ======================
st.sidebar.markdown('<p class="sidebar-title">📊 Dashboard Económico</p>', unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Ir a sección:",
    [
        "🏠 Carga de Datos",
        "🔎 EDA",
        "🔗 Análisis de Correlación",
        "📈 Análisis de Tendencia",
        "📊 Pivot Table"
    ]
)

st.sidebar.markdown('<p class="sidebar-footer">Hecho con ❤️ en Streamlit</p>', unsafe_allow_html=True)

# ======================
# 📌 Colores de fondo por sección
# ======================
bg_colors = {
    "🏠 Carga de Datos": "#F0F8FF",       # Azul claro
    "🔎 EDA": "#FAFAD2",                 # Amarillo pálido
    "🔗 Análisis de Correlación": "#E6E6FA", # Lavanda
    "📈 Análisis de Tendencia": "#FFE4E1",   # Rosa claro
    "📊 Pivot Table": "#F5F5DC"          # Beige
}

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_colors.get(menu, "#FFFFFF")};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# 📌 Secciones
# ======================

# --- CARGA DE DATOS ---
if menu == "🏠 Carga de Datos":
    st.header("📂 Carga de Datos")

    file = st.file_uploader("Sube tu archivo CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)

        # Intentar convertir columnas de fecha automáticamente
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        st.session_state.df = df
        st.success("✅ Datos cargados correctamente")
        st.write(df.head())

# --- EDA ---
elif menu == "🔎 EDA":
    st.header("🔎 Análisis Exploratorio de Datos (EDA)")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("Vista previa de los datos:")
        st.write(df.head())

        st.subheader("📊 Estadísticas descriptivas")
        st.write(df.describe(include="all"))

        st.subheader("📉 Distribuciones (Boxplot)")
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        if not num_cols.empty:
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_scaled, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No hay variables numéricas para mostrar boxplots.")
    else:
        st.warning("⚠️ Por favor carga primero un archivo CSV.")

# --- CORRELACIÓN ---
elif menu == "🔗 Análisis de Correlación":
    st.header("🔗 Análisis de Correlación")

    if st.session_state.df is not None:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns

        if len(num_cols) >= 2:
            var1 = st.selectbox("Selecciona la primera variable:", num_cols, index=0)
            var2 = st.selectbox("Selecciona la segunda variable:", num_cols, index=1)

            normalize = st.checkbox("Normalizar datos con MinMaxScaler")

            if var1 == var2:
                st.error("❌ No tiene sentido calcular la correlación con la misma variable.")
            else:
                data_corr = df[[var1, var2]].dropna()

                if normalize:
                    scaler = MinMaxScaler()
                    data_corr = pd.DataFrame(
                        scaler.fit_transform(data_corr),
                        columns=[var1, var2]
                    )

                corr_value = data_corr[var1].corr(data_corr[var2])

                st.success(f"📌 Índice de correlación de Pearson entre **{var1}** y **{var2}**: `{corr_value:.3f}`")

                fig, ax = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=data_corr, x=var1, y=var2, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("⚠️ Se requieren al menos dos variables numéricas para este análisis.")
    else:
        st.warning("⚠️ Por favor carga primero un archivo CSV.")

# --- TENDENCIA ---
elif menu == "📈 Análisis de Tendencia":
    st.header("📈 Análisis de Tendencia")

    if st.session_state.df is not None:
        df = st.session_state.df

        date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns

        if not date_cols.empty and not num_cols.empty:
            date_col = st.selectbox("Selecciona la columna de fecha:", date_cols)
            trend_var = st.selectbox("Selecciona la variable numérica:", num_cols)

            period = st.selectbox("Selecciona el periodo de resumen:",
                                  ["Día", "Semana", "Mes", "Trimestre", "Año"])

            freq_map = {"Día": "D", "Semana": "W", "Mes": "M", "Trimestre": "Q", "Año": "Y"}
            freq = freq_map[period]

            trend_data = df.groupby(pd.Grouper(key=date_col, freq=freq))[trend_var].mean().reset_index()

            st.line_chart(trend_data.set_index(date_col)[trend_var])
        else:
            st.warning("⚠️ Se requieren columnas de fecha y al menos una numérica.")
    else:
        st.warning("⚠️ Por favor carga primero un archivo CSV.")

# --- PIVOT TABLE ---
elif menu == "📊 Pivot Table":
    st.header("📊 Pivot Table")

    if st.session_state.df is not None:
        df = st.session_state.df

        date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns

        if not date_cols.empty and not cat_cols.empty and not num_cols.empty:
            date_col = st.selectbox("Selecciona la columna de fecha:", date_cols)
            cat_col = st.selectbox("Selecciona la variable categórica:", cat_cols)
            value_col = st.selectbox("Selecciona la variable numérica:", num_cols)

            pivot_table = pd.pivot_table(
                df,
                values=value_col,
                index=date_col,
                columns=cat_col,
                aggfunc="mean"
            )

            st.dataframe(pivot_table)
        else:
            st.warning("⚠️ El dataset necesita al menos: una columna de fecha, una categórica y una numérica.")
    else:
        st.warning("⚠️ Por favor carga primero un archivo CSV.")
