import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import numpy as np

# ===== NUEVO: LLM (Transformers + LangChain)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# ======================
# ESTILOS
# ======================
st.set_page_config(layout="wide", page_title="EDA Interactivo")

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
    .token-help {
        font-size: 12px; 
        color: #333; 
        margin-top: -10px; 
        margin-bottom: 10px;
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
     "Análisis de tendencias", "Pivot Table", "Análisis con LLM"]
)

# ======================
# CARGA DE DATOS + HF TOKEN/MODELO EN SIDEBAR
# ======================
if "df" not in st.session_state:
    st.session_state.df = None

if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""

if "llm" not in st.session_state:
    st.session_state.llm = None

if "hf_model_id" not in st.session_state:
    st.session_state.hf_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

if "use_4bit" not in st.session_state:
    st.session_state.use_4bit = True

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Hugging Face")
st.sidebar.text_input(
    "HF Token",
    type="password",
    key="hf_token",
    placeholder="hf_xxx...",
    help="Tu token de Hugging Face con acceso al modelo de Meta Llama 3."
)
st.sidebar.selectbox(
    "Modelo LLM",
    [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B"
    ],
    key="hf_model_id",
    help="Modelo recomendado: 8B Instruct."
)
st.sidebar.checkbox(
    "Cuantizar 4-bit (GPU)",
    value=True,
    key="use_4bit",
    help="Requiere CUDA + bitsandbytes; si falla se usará un modo alterno."
)

st.sidebar.markdown("---")
st.sidebar.info("App desarrollada con ❤️ usando Streamlit")

# ======================
# HELPERS LLM
# ======================
@st.cache_resource(show_spinner=True)
def load_llm(model_id: str, hf_token: str | None, use_4bit: bool = True) -> HuggingFacePipeline:
    """Carga el modelo de Hugging Face y lo expone como LLM de LangChain."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)

    # Intento de cuantización 4-bit si está habilitado
    model = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        except Exception:
            model = None

    # Fallback a FP16/FP32 si no hay 4-bit
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=text_gen)

def build_eda_context(df: pd.DataFrame, max_cols: int = 6, max_corr: int = 10) -> str:
    """Construye un contexto textual con hallazgos clave del EDA para el LLM."""
    lines = []
    lines.append(f"Tamaño del dataset: {df.shape[0]} filas x {df.shape[1]} columnas.")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    def preview_cols(cols):
        return cols[:max_cols] + (["..."] if len(cols) > max_cols else [])

    lines.append(f"Numéricas ({len(num_cols)}): {preview_cols(num_cols)}")
    lines.append(f"Categóricas ({len(cat_cols)}): {preview_cols(cat_cols)}")
    lines.append(f"Fechas ({len(date_cols)}): {preview_cols(date_cols)}")

    # Nulos
    nulls = df.isna().sum().sort_values(ascending=False)
    lines.append("Nulos (top): " + str(nulls.head(10).to_dict()))

    # Estadísticos numéricos
    if num_cols:
        desc = df[num_cols].describe().round(3).to_dict()
        limited_desc = {k: desc[k] for k in list(desc.keys())[:max_cols]}
        lines.append("Estadísticos (numéricas, parcial): " + str(limited_desc))

        # Correlaciones más fuertes (en valor absoluto)
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            corr_abs = corr.abs().where(~np.eye(len(corr), dtype=bool))
            pairs = corr_abs.unstack().dropna().sort_values(ascending=False)
            seen = set()
            top_pairs = {}
            for (a, b), v in pairs.items():
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                top_pairs[f"{a} ~ {b}"] = float(round(v, 3))
                if len(top_pairs) >= max_corr:
                    break
            lines.append("Correlaciones fuertes (|r|, top): " + str(top_pairs))

    return "\n".join(lines)

LLM_TEMPLATE = """Eres un analista de datos senior. Usa el contexto del EDA para responder con precisión y cautela.
Si la pregunta no es respondible con estos datos, dilo claramente y propone cómo investigarlo.

Contexto del EDA:
{context}

Pregunta del usuario:
{question}

Instrucciones:
- Responde en español claro.
- Cita columnas relevantes por nombre.
- Si mencionas correlaciones, indica el signo y posibles sesgos/causalidad.
- Reconoce limitaciones (nulos, tamaño de muestra, atípicos).
- Cierra con 1–2 siguientes pasos prácticos.

Respuesta:
"""
LLM_PROMPT = PromptTemplate.from_template(LLM_TEMPLATE)

# ======================
# SECCIÓN: CARGA DE DATOS
# ======================
if menu == "Carga de datos":
    st.markdown("<div style='background-color:#f0f8ff;padding:20px;border-radius:10px'>", unsafe_allow_html=True)
    st.header("📂 Carga de datos")

    file = st.file_uploader("Sube tu archivo CSV", type="csv")

    st.markdown(
        "<div class='token-help'>Para usar el LLM, ingresa tu token de Hugging Face arriba en el sidebar y acepta la licencia del modelo en Hugging Face.</div>",
        unsafe_allow_html=True
    )

    if file:
        try:
            # Se carga el DataFrame sin conversiones iniciales
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin1")

        # --- Detección generalista y robusta de tipos ---
        for col in df.columns:
            # 1) Fecha si >70% convierte a datetime
            try:
                converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if converted.notna().mean() > 0.7:
                    df[col] = converted
                    continue
            except Exception:
                pass
            # 2) Numérico si >70% convierte a número
            if df[col].dtype == 'object':
                try:
                    numeric_try = pd.to_numeric(df[col].str.replace(",", "").str.replace("$", ""), errors="coerce")
                except Exception:
                    numeric_try = pd.to_numeric(df[col], errors="coerce")
                if numeric_try.notna().mean() > 0.7 and df[col].nunique(dropna=True) > 5:
                    df[col] = numeric_try
                    continue
            # 3) Dejar como categórica si no cumple

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
# NULOS Y ATÍPICOS
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
            num_cols = df.select_dtypes(include=np.number).columns
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(num_cols) > 0:
                for col in num_cols:
                    if num_imputation_method == "Media":
                        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                    elif num_imputation_method == "Mediana":
                        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                    elif num_imputation_method == "Moda":
                        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else df_copy[col])
            
            if len(cat_cols) > 0:
                for col in cat_cols:
                    if cat_imputation_method == "Moda":
                        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else "Desconocido")
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
        st.subheader("Matriz de correlación (verde=positiva, rojo=negativa)")
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
# ANÁLISIS CON LLM (Llama 3 + HF + LangChain)
# ======================
if menu == "Análisis con LLM":
    st.header("🤖 Análisis con LLM (Llama 3 Instruct)")
    if st.session_state.df is None:
        st.warning("Primero carga un CSV en la sección 'Carga de datos' y proporciona tu HF Token en el sidebar.")
    else:
        df = st.session_state.df

        with st.expander("🔎 Ver resumen del EDA que se enviará al modelo", expanded=False):
            eda_context = build_eda_context(df)
            st.text(eda_context)

        # Botón de carga de modelo
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("🚀 Cargar / actualizar modelo"):
                if not st.session_state.hf_token:
                    st.error("Necesitas proporcionar un HF token válido en el sidebar.")
                else:
                    with st.spinner("Cargando modelo..."):
                        try:
                            st.session_state.llm = load_llm(
                                model_id=st.session_state.hf_model_id,
                                hf_token=st.session_state.hf_token,
                                use_4bit=st.session_state.use_4bit
                            )
                            st.success("Modelo listo ✅")
                        except Exception as e:
                            st.error(f"No se pudo cargar el modelo: {e}")

        # Interacción con el LLM
        if st.session_state.llm is not None:
            st.subheader("Haz preguntas sobre tu dataset")
            user_q = st.text_area("Pregunta", placeholder="Ej.: ¿Qué variables están más correlacionadas con el objetivo?")
            ask = st.button("Preguntar al LLM")

            if ask:
                if not user_q.strip():
                    st.warning("Escribe una pregunta primero.")
                else:
                    chain = LLM_PROMPT | st.session_state.llm
                    with st.spinner("Generando respuesta..."):
                        try:
                            answer = chain.invoke({"context": eda_context, "question": user_q})
                            st.markdown("### Respuesta del modelo")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Ocurrió un error al generar la respuesta: {e}")

            # Historial simple de conversación
            st.markdown("---")
            if "chat" not in st.session_state:
                st.session_state.chat = []
            if ask and user_q.strip():
                st.session_state.chat.append(("usuario", user_q))
                if 'answer' in locals():
                    st.session_state.chat.append(("modelo", answer))

            if st.checkbox("Mostrar historial de conversación"):
                for role, msg in st.session_state.chat:
                    st.markdown(f"**{role.capitalize()}:** {msg}")
