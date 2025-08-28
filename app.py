import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import numpy as np

# 🔹 Integración con LLM
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ======================
# ESTILOS
# ======================
st.set_page_config(layout="wide", page_title="EDA Interactivo con LLM")

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
     "Análisis de tendencias", "Pivot Table", "Asistente LLM"]
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

        # --- Mejora en detección de tipos ---
        for col in df.columns:
            if "date" in col.lower() or "fecha" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    st.write(f"✅ Columna '{col}' reconocida y convertida a tipo fecha.")
                except Exception as e:
                    st.error(f"❌ Error al convertir la columna '{col}' a fecha: {e}")
        for col in df.columns:
            if df[col].dtype == 'object':
                temp_series = pd.to_numeric(df[col], errors='coerce')
                if (temp_series.notnull().sum() / len(df)) > 0.9 and df[col].nunique() > 10:
                    df[col] = temp_series
                    st.write(f"✅ Columna '{col}' convertida a tipo numérico.")
        # --- Fin mejora ---

        st.session_state.df = df
        st.success("✅ Datos cargados y tipos de datos detectados correctamente")
        st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

# ======================
# DESCRIPCIÓN GENERAL
# ======================
if menu == "Descripción general" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📖 Descripción general")
    st.write("### Tipos de datos detectados")
    st.dataframe(df.dtypes, use_container_width=True)

    st.write("### Resumen estadístico (numéricas)")
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        st.write(df[num_cols].describe())
    else:
        st.warning("No se detectaron variables numéricas.")

# ======================
# ANÁLISIS DE VALORES NULOS Y ATÍPICOS
# ======================
if menu == "Análisis de valores nulos y atípicos" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("🔎 Valores nulos y atípicos")

    st.subheader("Valores nulos por columna")
    st.write(df.isnull().sum())

    st.markdown("---")
    st.subheader("🛠️ Gestión de valores nulos")
    
    missing_strategy = st.radio(
        "Elige una estrategia para manejar los valores nulos:",
        ("No hacer nada", "Eliminar filas", "Imputar valores"),
        horizontal=True
    )
    
    if missing_strategy == "Imputar valores":
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        
        st.markdown("#### Columnas numéricas")
        num_imputation_method = st.selectbox(
            "Método de imputación para variables numéricas:",
            ("Mediana", "Media", "Moda")
        )
        
        st.markdown("#### Columnas categóricas")
        cat_imputation_method = st.radio(
            "Método de imputación para variables categóricas:",
            ("Moda", "Valor fijo 'Desconocido'"),
            horizontal=True
        )

    if st.button("Aplicar cambios"):
        df_copy = df.copy()

        if missing_strategy == "Eliminar filas":
            df_copy = df_copy.dropna()
            st.success("✅ Filas con valores nulos eliminadas correctamente.")
        
        elif missing_strategy == "Imputar valores":
            if len(num_cols) > 0:
                for col in num_cols:
                    if num_imputation_method == "Media":
                        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                    elif num_imputation_method == "Mediana":
                        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                    elif num_imputation_method == "Moda":
                        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            
            if len(cat_cols) > 0:
                for col in cat_cols:
                    if cat_imputation_method == "Moda":
                        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
                    elif cat_imputation_method == "Valor fijo 'Desconocido'":
                        df_copy[col] = df_copy[col].fillna("Desconocido")
            
            st.success("✅ Valores nulos imputados correctamente.")

        st.session_state.df = df_copy
        st.write("### Nuevos valores nulos por columna:")
        st.write(st.session_state.df.isnull().sum())

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        st.subheader("Valores atípicos (Z-score > 3)")
        outliers = (df[num_cols].apply(lambda x: zscore(x, nan_policy='omit')).abs() > 3).sum()
        st.write(outliers)

# ======================
# VISUALIZACIÓN NUMÉRICA
# ======================
if menu == "Visualización numérica" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Boxplots de variables numéricas")

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        normalize = st.checkbox("Normalizar datos con MinMaxScaler", value=False)
        data_plot = df[num_cols].copy()
        
        if normalize:
            scaler = MinMaxScaler()
            data_plot = pd.DataFrame(scaler.fit_transform(data_plot), columns=num_cols)

        fig, axes = plt.subplots(nrows=(len(num_cols) + 1) // 2, ncols=2, figsize=(12, 6))
        if len(num_cols) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i, col in enumerate(num_cols):
            sns.boxplot(y=data_plot[col].dropna(), ax=axes[i], color="skyblue")
            axes[i].set_title(col)
        
        for i in range(len(num_cols), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No se encontraron columnas numéricas para visualizar.")

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
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("No se encontraron columnas categóricas para visualizar.")

# ======================
# CORRELACIONES
# ======================
if menu == "Correlaciones" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Correlaciones entre variables")

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 1:
        st.subheader("Matriz de correlación")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlación entre dos variables")
        normalize_corr = st.checkbox("Normalizar antes de correlacionar", value=False)

        col1, col2 = st.columns(2)
        var1 = col1.selectbox("Variable 1", num_cols, index=0)
        var2 = col2.selectbox("Variable 2", num_cols, index=1 if len(num_cols) > 1 else 0)

        if var1 == var2:
            st.error("❌ No tiene sentido correlacionar la misma variable.")
        else:
            data_corr = df[[var1, var2]].dropna()
            if data_corr.empty:
                st.warning("No hay suficientes datos limpios para correlacionar estas variables.")
            else:
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
    num_cols = df.select_dtypes(include=np.number).columns

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
    num_cols = df.select_dtypes(include=np.number).columns

    if len(date_cols) > 0 and len(cat_cols) > 0 and len(num_cols) > 0:
        date_col = st.selectbox("Selecciona columna de fecha", date_cols)
        cat_col = st.selectbox("Selecciona columna categórica (ej. Stock Index)", cat_cols)
        num_var = st.selectbox("Selecciona variable numérica", num_cols)

        pivot = pd.pivot_table(df.dropna(subset=[date_col, cat_col, num_var]), 
                               index=date_col, columns=cat_col, values=num_var, aggfunc="mean")

        st.dataframe(pivot.head())
    else:
        st.warning("Se necesitan al menos una columna de fecha, una categórica y una numérica.")

# ======================
# ASISTENTE LLM
# ======================
if menu == "Asistente LLM" and st.session_state.df is not None:
    st.header("🤖 Asistente LLM sobre tu dataset")

    # --- Persistencia de API Key en la sesión ---
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = None

    if st.session_state.groq_api_key:
        st.info("🔑 API Key cargada en esta sesión.")
    else:
        groq_api_key = st.text_input("Ingresa tu API Key de Groq", type="password")
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
            st.success("✅ API Key guardada en esta sesión")

    # Botón para cerrar sesión de la API Key
    if st.session_state.groq_api_key:
        if st.button("Cerrar sesión de API Key"):
            st.session_state.groq_api_key = None
            st.info("🔒 API Key eliminada de la sesión.")

    # --- Si hay token, inicializamos el modelo ---
    if st.session_state.groq_api_key:
        model = ChatGroq(
            groq_api_key=st.session_state.groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )

        # Prompt Template
        template = """
        Eres un experto en ciencia de datos.
        Dataset cargado con las siguientes columnas: {columns}.
        Resumen estadístico:
        {stats}

        El usuario pregunta: {question}
        Responde en español de manera clara y concisa.
        """
        prompt = PromptTemplate(
            input_variables=["columns", "stats", "question"],
            template=template
        )
        chain = LLMChain(llm=model, prompt=prompt)

        user_question = st.text_input("Escribe tu pregunta sobre el dataset:")
        if user_question:
            df = st.session_state.df
            columns = ", ".join(df.columns)
            stats = df.describe(include="all").to_string()

            response = chain.run(columns=columns, stats=stats, question=user_question)
            st.markdown("### 📌 Respuesta del asistente:")
            st.write(response)
    else:
        st.warning("⚠️ Ingresa tu API Key para usar el asistente.")
