import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# --- Custom CSS para diseño con dos frames ---
st.markdown("""
    <style>
    /* Dividir en columnas */
    .mainframe {
        display: flex;
        flex-direction: row;
    }
    /* Frame izquierdo (menú) */
    .sidebar-custom {
        background-color: #e6f0fa; /* Azul claro */
        width: 25%;
        min-height: 100vh;
        padding: 20px;
        border-radius: 0 20px 20px 0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.1);
    }
    /* Frame derecho (contenido principal) */
    .content-custom {
        flex-grow: 1;
        padding: 20px;
        border-radius: 20px;
        margin-left: 10px;
    }
    /* Títulos del menú */
    .menu-title {
        font-size: 22px;
        font-weight: bold;
        color: #003366;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Variables globales ---
if "df" not in st.session_state:
    st.session_state.df = None

# --- Estructura principal ---
st.markdown('<div class="mainframe">', unsafe_allow_html=True)

# Frame izquierdo (menú)
st.markdown('<div class="sidebar-custom">', unsafe_allow_html=True)
st.markdown('<div class="menu-title">📊 Menú de Navegación</div>', unsafe_allow_html=True)

menu = st.radio(
    "Ir a sección:",
    ["Carga de Datos", "EDA", "Análisis de Correlación", "Análisis de Tendencia", "Pivot Table"],
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)  # cierre sidebar

# =======================
# Frame derecho (contenido principal)
# =======================

# --- CARGA DE DATOS ---
if menu == "Carga de Datos":
    st.markdown('<div class="content-custom" style="background-color:#fef9e7;">', unsafe_allow_html=True)
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
        st.success("Datos cargados correctamente ✅")
        st.write(df.head())

    st.markdown('</div>', unsafe_allow_html=True)

# --- EDA ---
elif menu == "EDA":
    st.markdown('<div class="content-custom" style="background-color:#e8f8f5;">', unsafe_allow_html=True)
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
            st.warning("No hay variables numéricas para mostrar boxplots.")
    else:
        st.warning("Por favor carga primero un archivo CSV.")

    st.markdown('</div>', unsafe_allow_html=True)

# --- CORRELACIÓN ---
elif menu == "Análisis de Correlación":
    st.markdown('<div class="content-custom" style="background-color:#fce4ec;">', unsafe_allow_html=True)
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

                st.write(f"📌 **Índice de correlación de Pearson entre {var1} y {var2}: {corr_value:.3f}**")

                fig, ax = plt.subplots(figsize=(7, 5))
                sns.scatterplot(data=data_corr, x=var1, y=var2, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Se requieren al menos dos variables numéricas para este análisis.")
    else:
        st.warning("Por favor carga primero un archivo CSV.")

    st.markdown('</div>', unsafe_allow_html=True)

# --- TENDENCIA ---
elif menu == "Análisis de Tendencia":
    st.markdown('<div class="content-custom" style="background-color:#e3f2fd;">', unsafe_allow_html=True)
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
            st.warning("Se requieren columnas de fecha y al menos una numérica.")
    else:
        st.warning("Por favor carga primero un archivo CSV.")

    st.markdown('</div>', unsafe_allow_html=True)

# --- PIVOT TABLE ---
elif menu == "Pivot Table":
    st.markdown('<div class="content-custom" style="background-color:#f9fbe7;">', unsafe_allow_html=True)
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
            st.warning("El dataset necesita al menos: una columna de fecha, una categórica y una numérica.")
    else:
        st.warning("Por favor carga primero un archivo CSV.")

    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# Cierre del mainframe
# =======================
st.markdown('</div>', unsafe_allow_html=True)
