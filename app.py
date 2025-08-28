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
# ESTILOS / PAGE CONFIG
# ======================
st.set_page_config(layout="wide", page_title="EDA Interactivo con LLM")

st.markdown("""
    <style>
    .sidebar .sidebar-content { background-color: #e6f0ff; }
    .block-container { max-width: 1300px; padding: 1.2rem 2rem; }
    h1, h2, h3 { color: #003366; }
    </style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio(
    "📊 Menú de navegación",
    [
        "Carga de datos",
        "Descripción general",
        "Análisis de valores nulos y atípicos",
        "Visualización numérica",
        "Visualización categórica",
        "Correlaciones",
        "Análisis de tendencias",
        "Pivot Table",
        "Asistente LLM",
        "📖 Documentación"
    ]
)

# ======================
# SESSION STATE
# ======================
if "df" not in st.session_state:
    st.session_state.df = None
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = None

st.sidebar.markdown("---")
st.sidebar.info("App desarrollada con ❤️ usando Streamlit")

# ======================
# CARGA DE DATOS
# ======================
if menu == "Carga de datos":
    st.header("📂 Carga de datos")
    file = st.file_uploader("Sube tu archivo CSV", type="csv")

    if file:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1")

        # --- Detección y conversión de fechas ---
        for col in df.columns:
            if "date" in col.lower() or "fecha" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    st.write(f"✅ Columna '{col}' convertida a fecha.")
                except Exception as e:
                    st.warning(f"⚠️ No se pudo convertir '{col}' a fecha: {e}")

        # --- Conversión de objetos a numérico si aplica ---
        for col in df.columns:
            if df[col].dtype == "object":
                tmp = pd.to_numeric(df[col], errors="coerce")
                if (tmp.notnull().sum() / len(df) > 0.9) and df[col].nunique() > 10:
                    df[col] = tmp
                    st.write(f"✅ Columna '{col}' convertida a numérico.")

        st.session_state.df = df
        st.success("✅ Datos cargados correctamente")
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.info("Carga un CSV para comenzar.")

# ======================
# DESCRIPCIÓN GENERAL
# ======================
elif menu == "Descripción general" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📖 Descripción general")
    st.subheader("Tipos de datos")
    st.dataframe(df.dtypes, use_container_width=True)

    st.subheader("Resumen estadístico (numéricas)")
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        st.write(df[num_cols].describe())
    else:
        st.warning("No se detectaron variables numéricas.")

# ======================
# ANÁLISIS DE VALORES NULOS Y ATÍPICOS
# ======================
elif menu == "Análisis de valores nulos y atípicos" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("🔎 Valores nulos y atípicos")

    st.subheader("Valores nulos por columna")
    st.write(df.isnull().sum())

    st.markdown("---")
    st.subheader("🛠️ Gestión de valores nulos")
    missing_strategy = st.radio(
        "Estrategia:",
        ("No hacer nada", "Eliminar filas", "Imputar valores"),
        horizontal=True
    )

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if missing_strategy == "Imputar valores":
        st.markdown("#### Columnas numéricas")
        num_method = st.selectbox("Método numérico:", ("Mediana", "Media", "Moda"))
        st.markdown("#### Columnas categóricas")
        cat_method = st.radio("Método categórico:", ("Moda", "Valor fijo 'Desconocido'"), horizontal=True)

    if st.button("Aplicar cambios"):
        df_copy = df.copy()

        if missing_strategy == "Eliminar filas":
            df_copy = df_copy.dropna()
            st.success("✅ Filas con valores nulos eliminadas.")

        elif missing_strategy == "Imputar valores":
            if len(num_cols) > 0:
                for c in num_cols:
                    if num_method == "Media":
                        df_copy[c] = df_copy[c].fillna(df_copy[c].mean())
                    elif num_method == "Mediana":
                        df_copy[c] = df_copy[c].fillna(df_copy[c].median())
                    else:
                        df_copy[c] = df_copy[c].fillna(df_copy[c].mode()[0])
            if len(cat_cols) > 0:
                for c in cat_cols:
                    if cat_method == "Moda":
                        df_copy[c] = df_copy[c].fillna(df_copy[c].mode()[0])
                    else:
                        df_copy[c] = df_copy[c].fillna("Desconocido")
            st.success("✅ Valores nulos imputados.")

        st.session_state.df = df_copy
        st.write("### Nuevos nulos por columna")
        st.write(st.session_state.df.isnull().sum())

    if len(num_cols) > 0:
        st.subheader("Valores atípicos (Z-score > 3)")
        outliers = (df[num_cols].apply(lambda x: zscore(x, nan_policy='omit')).abs() > 3).sum()
        st.write(outliers)

# ======================
# VISUALIZACIÓN NUMÉRICA
# ======================
elif menu == "Visualización numérica" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Boxplots de variables numéricas")

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        normalize = st.checkbox("Normalizar con MinMaxScaler", value=False)
        data_plot = df[num_cols].copy()
        if normalize:
            scaler = MinMaxScaler()
            data_plot = pd.DataFrame(scaler.fit_transform(data_plot), columns=num_cols)

        rows = (len(num_cols) + 1) // 2
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, 6))
        axes = np.array(axes).reshape(-1) if len(num_cols) > 1 else [axes]
        for i, col in enumerate(num_cols):
            sns.boxplot(y=data_plot[col].dropna(), ax=axes[i], color="skyblue")
            axes[i].set_title(col)
        for j in range(len(num_cols), len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No hay columnas numéricas para visualizar.")

# ======================
# VISUALIZACIÓN CATEGÓRICA
# ======================
elif menu == "Visualización categórica" and st.session_state.df is not None:
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
        st.warning("No hay columnas categóricas para visualizar.")

# ======================
# CORRELACIONES
# ======================
elif menu == "Correlaciones" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Correlaciones")
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
                st.warning("No hay suficientes datos limpios para correlacionar.")
            else:
                if normalize_corr:
                    scaler = MinMaxScaler()
                    data_corr = pd.DataFrame(scaler.fit_transform(data_corr), columns=[var1, var2])
                st.info(f"Coeficiente de Pearson: **{data_corr[var1].corr(data_corr[var2]):.4f}**")
    else:
        st.warning("Se requieren al menos 2 columnas numéricas.")

# ======================
# ANÁLISIS DE TENDENCIAS (RESTABLECIDO Y MEJORADO)
# ======================
elif menu == "Análisis de tendencias" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📈 Análisis de tendencias")

    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(date_cols) == 0 or len(num_cols) == 0:
        st.warning("Necesitas al menos una **columna de fecha** y una **numérica**.")
    else:
        c1, c2, c3 = st.columns([1.1, 1, 1])
        date_col = c1.selectbox("Columna de fecha", date_cols, index=0)
        trend_var = c2.selectbox("Variable numérica", num_cols, index=0)

        period = c3.radio("Frecuencia", ["Día", "Semana", "Mes", "Trimestre", "Año"], horizontal=True)
        freq_map = {"Día": "D", "Semana": "W", "Mes": "M", "Trimestre": "Q", "Año": "Y"}
        freq = freq_map[period]

        agg = st.selectbox("Agregación", ["Media", "Suma", "Mediana"], index=0)
        agg_map = {"Media": "mean", "Suma": "sum", "Mediana": "median"}
        agg_fn = agg_map[agg]

        fill_gaps = st.checkbox("Rellenar huecos temporales con interpolación", value=False)
        show_ma = st.checkbox("Mostrar media móvil", value=True)
        ma_window = st.slider("Ventana media móvil (periodos)", min_value=2, max_value=60, value=6)
        show_outliers = st.checkbox("Resaltar atípicos por Z-score", value=True)
        z_thr = st.slider("Umbral Z-score", min_value=2.0, max_value=5.0, value=3.0, step=0.1)

        # --- Serie principal ---
        ts = (
            df.dropna(subset=[date_col, trend_var])
              .set_index(date_col)
              .sort_index()[trend_var]
              .resample(freq)
              .agg(agg_fn)
        )

        if fill_gaps:
            ts = ts.asfreq(freq)
            ts = ts.interpolate(method="time")

        trend_df = ts.reset_index().rename(columns={trend_var: "value"})

        # Plot principal
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=trend_df, x=date_col, y="value", ax=ax)
        ax.set_title(f"{trend_var} por {period} ({agg})")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(trend_var)

        # Media móvil
        if show_ma and len(trend_df) >= ma_window:
            ma = trend_df["value"].rolling(window=ma_window, min_periods=max(1, ma_window//2)).mean()
            ax.plot(trend_df[date_col], ma, linewidth=2)

        # Atípicos por Z-score
        if show_outliers and trend_df["value"].std(ddof=0) > 0:
            z = (trend_df["value"] - trend_df["value"].mean()) / trend_df["value"].std(ddof=0)
            mask = z.abs() > z_thr
            ax.scatter(trend_df.loc[mask, date_col], trend_df.loc[mask, "value"], s=40)

        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Comparación por categoría (opcional)")
        if len(cat_cols) > 0:
            ccol1, ccol2 = st.columns([1.2, 1])
            cat_col = ccol1.selectbox("Columna categórica", ["(Ninguna)"] + list(cat_cols), index=0)
            top_n = ccol2.slider("Top-N categorías por frecuencia", 2, 12, 6)

            if cat_col != "(Ninguna)":
                top_cats = (
                    df[cat_col].value_counts(dropna=True)
                    .head(top_n)
                    .index.tolist()
                )
                dff = df[df[cat_col].isin(top_cats)].dropna(subset=[date_col, trend_var, cat_col])

                comp = (
                    dff.set_index(date_col)
                       .groupby([pd.Grouper(freq=freq), cat_col])[trend_var]
                       .agg(agg_fn)
                       .reset_index()
                )

                fig2, ax2 = plt.subplots(figsize=(11, 5))
                sns.lineplot(data=comp, x=date_col, y=trend_var, hue=cat_col, ax=ax2, marker="o")
                ax2.set_title(f"{trend_var} por {period} ({agg}) - Top {top_n} {cat_col}")
                ax2.set_xlabel("Fecha")
                ax2.set_ylabel(trend_var)
                st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Descomposición estacional (opcional)")
        do_decomp = st.checkbox("Intentar descomposición estacional (statsmodels)", value=False)
        if do_decomp:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                # Asegurar índice de fecha con frecuencia
                ts_decomp = ts.asfreq(freq)
                ts_decomp = ts_decomp.interpolate(method="time")
                if ts_decomp.isna().all() or ts_decomp.nunique() < 3:
                    st.warning("No hay suficiente variabilidad o datos para descomponer.")
                else:
                    model = "additive"
                    result = seasonal_decompose(ts_decomp, model=model, period=None, two_sided=True, extrapolate_trend="freq")
                    fig3 = result.plot()
                    fig3.set_size_inches(11, 7)
                    st.pyplot(fig3)
            except Exception as e:
                st.warning(f"No se pudo realizar la descomposición. Instala statsmodels o revisa los datos. Detalle: {e}")

# ======================
# PIVOT TABLE
# ======================
elif menu == "Pivot Table" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("📊 Pivot Table con promedio")

    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=np.number).columns

    if len(date_cols) > 0 and len(cat_cols) > 0 and len(num_cols) > 0:
        date_col = st.selectbox("Columna de fecha", date_cols)
        cat_col = st.selectbox("Columna categórica (ej. índice/segmento)", cat_cols)
        num_var = st.selectbox("Variable numérica", num_cols)

        pivot = pd.pivot_table(
            df.dropna(subset=[date_col, cat_col, num_var]),
            index=date_col, columns=cat_col, values=num_var, aggfunc="mean"
        )
        st.dataframe(pivot.head(), use_container_width=True)
    else:
        st.warning("Se necesitan al menos una columna de fecha, una categórica y una numérica.")

# ======================
# ASISTENTE LLM (TOKEN PERSISTENTE)
# ======================
elif menu == "Asistente LLM" and st.session_state.df is not None:
    st.header("🤖 Asistente LLM sobre tu dataset")

    if st.session_state.groq_api_key:
        st.info("🔑 API Key cargada en esta sesión.")
    else:
        groq_api_key = st.text_input("Ingresa tu API Key de Groq", type="password")
        if groq_api_key:
            st.session_state.groq_api_key = groq_api_key
            st.success("✅ API Key guardada en esta sesión")

    if st.session_state.groq_api_key:
        if st.button("Cerrar sesión de API Key"):
            st.session_state.groq_api_key = None
            st.info("🔒 API Key eliminada de la sesión.")

        model = ChatGroq(
            groq_api_key=st.session_state.groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )

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
            # describe(include="all") puede ser grande; limitar filas/cols si lo deseas
            stats = df.describe(include="all").transpose().fillna("").astype(str).head(100).to_string()

            response = chain.run(columns=columns, stats=stats, question=user_question)
            st.markdown("### 📌 Respuesta del asistente:")
            st.write(response)
    else:
        st.warning("⚠️ Ingresa tu API Key para usar el asistente.")

# ======================
# DOCUMENTACIÓN
# ======================
elif menu == "📖 Documentación":
    st.header("📖 Documentación del Asistente LLM y Módulo de Tendencias")

    st.markdown("""
    ## 🔹 Descripción general
    Esta app facilita el **EDA** y agrega un **Asistente LLM** (via Groq + LangChain) que contesta preguntas
    sobre tu dataset. El módulo de **Tendencias** permite analizar series temporales con resampleo,
    agregaciones, suavizado, detección de atípicos y comparaciones por categoría.

    ## 🔹 Arquitectura del LLM
    - **Modelo:** `llama-3.3-70b-versatile`
    - **Proveedor:** Groq (baja latencia)
    - **Integración:** `langchain_groq.ChatGroq` + `LLMChain` con `PromptTemplate`
    - **Contexto al modelo:** columnas + `describe()` resumido + pregunta del usuario

    ## 🔹 Módulo de Tendencias (cómo funciona)
    1. Seleccionas **columna de fecha** y **variable numérica**.
    2. Eliges **frecuencia** (D/W/M/Q/Y) y **agregación** (media, suma, mediana).
    3. (Opcional) Rellenas huecos e interpolas.
    4. (Opcional) Añades **media móvil** y resaltas **atípicos** por Z-score.
    5. (Opcional) Comparas por **categoría** (Top-N líneas).
    6. (Opcional) **Descomposición estacional** si tienes `statsmodels`.

    ## 🔹 Buenas prácticas
    - Normaliza formatos de fecha antes de cargar si puedes.
    - Para comparaciones por categoría, limita a Top-N para evitar gráficos saturados.
    - Ajusta la **ventana** de media móvil según la frecuencia (p.ej., 7 para diario, 3 para mensual).

    ## 🔹 Limitaciones
    - El LLM no ejecuta consultas SQL ni cálculos adicionales fuera del contexto entregado.
    - Series muy cortas o con muchos huecos pueden afectar la descomposición y el Z-score.
    """)

# ======================
# GUARDAS: DF NO CARGADO
# ======================
else:
    if st.session_state.df is None:
        st.info("Carga un dataset en **Carga de datos** para habilitar las demás secciones.")
