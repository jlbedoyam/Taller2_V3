import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Aplicación de EDA con Streamlit", layout="wide")

# --- Estilos CSS ---
st.markdown("""
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #cce7ff;
    }
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-h5rgaw {
        color: #003366;
        font-size: 18px;
    }
    .section-header {
        font-size:22px;
        font-weight:600;
        margin-bottom:10px;
    }
    .credits {
        font-size: 12px;
        color: #555;
        text-align: center;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Menú lateral ---
st.sidebar.title("📊 Menú de navegación")
menu = st.sidebar.radio("Ir a sección:", [
    "Carga de datos",
    "Descripción general",
    "Visualización de variables",
    "Correlaciones",
    "Análisis de tendencia",
    "Tabla dinámica"
])

# --- Créditos al final del sidebar ---
st.sidebar.markdown("<div class='credits'>Creado con ❤️ usando Streamlit</div>", unsafe_allow_html=True)

# --- Carga de datos ---
if menu == "Carga de datos":
    st.markdown("<div class='section-header'>📂 Carga de datos</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.session_state["df"] = df

# --- Descripción general ---
elif menu == "Descripción general":
    st.markdown("<div class='section-header'>ℹ️ Descripción general</div>", unsafe_allow_html=True)
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.write("Dimensiones del dataset:", df.shape)
        st.write("Tipos de datos:")
        st.write(df.dtypes)
        st.write("Valores nulos por columna:")
        st.write(df.isnull().sum())
    else:
        st.warning("Primero carga un archivo en la sección 'Carga de datos'.")

# --- Visualización de variables ---
elif menu == "Visualización de variables":
    st.markdown("<div class='section-header'>📈 Visualización de variables</div>", unsafe_allow_html=True)
    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include="object").columns

        # Boxplots
        normalize = st.checkbox("Normalizar con MinMaxScaler antes del boxplot")
        if normalize:
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
        else:
            df_scaled = df[numeric_cols]

        if not df_scaled.empty:
            st.write("### Boxplots variables numéricas")
            fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(6, 4 * len(numeric_cols)))
            if len(numeric_cols) == 1:
                axes = [axes]
            for i, col in enumerate(numeric_cols):
                sns.boxplot(data=df_scaled, x=col, ax=axes[i])
                axes[i].set_title(f"Boxplot de {col}")
            st.pyplot(fig)

        # Histogramas categóricos
        if len(cat_cols) > 0:
            st.write("### Histogramas variables categóricas")
            for col in cat_cols:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Frecuencia de {col}")
                st.pyplot(fig)
    else:
        st.warning("Primero carga un archivo en la sección 'Carga de datos'.")

# --- Correlaciones ---
elif menu == "Correlaciones":
    st.markdown("<div class='section-header'>🔗 Análisis de correlaciones</div>", unsafe_allow_html=True)
    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 1:
            # Heatmap
            st.write("### Heatmap de correlaciones")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, cmap="RdYlGn", center=0, annot=True, ax=ax)
            st.pyplot(fig)

            # Correlación entre dos variables
            st.write("### Correlación entre dos variables numéricas")
            var1 = st.selectbox("Selecciona la primera variable", numeric_cols, index=0)
            var2 = st.selectbox("Selecciona la segunda variable", numeric_cols, index=1)
            normalize_corr = st.checkbox("Normalizar con MinMaxScaler", key="corr_norm")

            if var1 == var2:
                st.error("⚠️ No tiene sentido correlacionar una variable consigo misma.")
            else:
                data_corr = df[[var1, var2]].dropna()
                if normalize_corr:
                    scaler = MinMaxScaler()
                    data_corr[[var1, var2]] = scaler.fit_transform(data_corr[[var1, var2]])

                corr_value = data_corr[var1].corr(data_corr[var2], method="pearson")
                st.write(f"Coeficiente de correlación de **Pearson** entre {var1} y {var2}: `{corr_value:.2f}`")

                fig, ax = plt.subplots()
                sns.scatterplot(data=data_corr, x=var1, y=var2, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("No hay suficientes variables numéricas para calcular correlaciones.")
    else:
        st.warning("Primero carga un archivo en la sección 'Carga de datos'.")

# --- Análisis de tendencia ---
elif menu == "Análisis de tendencia":
    st.markdown("<div class='section-header'>📉 Análisis de tendencia</div>", unsafe_allow_html=True)
    if "df" in st.session_state:
        df = st.session_state["df"]
        date_cols = df.select_dtypes(include="datetime").columns
        if not list(date_cols):
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols = [col]
                    break
                except:
                    pass
        if date_cols:
            date_col = st.selectbox("Selecciona la columna de fecha", date_cols)
            numeric_cols = df.select_dtypes(include=np.number).columns
            trend_var = st.selectbox("Selecciona variable numérica a analizar", numeric_cols)
            period = st.selectbox("Periodo de resumen", ["Día", "Mes", "Trimestre", "Año"])

            freq_map = {"Día": "D", "Mes": "M", "Trimestre": "Q", "Año": "Y"}
            trend_data = df.groupby(pd.Grouper(key=date_col, freq=freq_map[period]))[trend_var].mean().reset_index()

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=trend_data, x=date_col, y=trend_var, ax=ax)
            ax.set_title(f"Tendencia de {trend_var} por {period}")
            st.pyplot(fig)
        else:
            st.warning("No se encontró columna de tipo fecha en el dataset.")
    else:
        st.warning("Primero carga un archivo en la sección 'Carga de datos'.")

# --- Tabla dinámica ---
elif menu == "Tabla dinámica":
    st.markdown("<div class='section-header'>📊 Tabla dinámica</div>", unsafe_allow_html=True)
    if "df" in st.session_state:
        df = st.session_state["df"]
        if "Stock Index" in df.columns:
            date_cols = df.select_dtypes(include="datetime").columns
            if not list(date_cols):
                for col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_cols = [col]
                        break
                    except:
                        pass
            if date_cols:
                date_col = date_cols[0]
                pivot = pd.pivot_table(
                    df,
                    values=df.select_dtypes(include=np.number).columns,
                    index=date_col,
                    columns="Stock Index",
                    aggfunc="mean"
                )
                st.dataframe(pivot)
            else:
                st.warning("No se encontró columna de fecha para la tabla dinámica.")
        else:
            st.warning("No existe la columna 'Stock Index' en el dataset.")
    else:
        st.warning("Primero carga un archivo en la sección 'Carga de datos'.")
