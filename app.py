import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ====== Configuración de página y estilo ======
st.set_page_config(page_title="EDA Automático", layout="wide")
st.markdown("""
<style>
    .main { background-color: #f7f8fa; }
    h1, h2, h3 { color: #2c3e50; }
    .stDataFrame { background: #fff; border-radius: 10px; }
    .metric-card { background:#fff; padding:12px 16px; border-radius:12px; border:1px solid #eaecef; }
</style>
""", unsafe_allow_html=True)
st.title("📊 Exploratory Data Analysis (EDA) – Interactivo")

# ====== Utilidades ======
def deduplicate_columns(df: pd.DataFrame):
    """Renombra columnas duplicadas añadiendo sufijos _dup1, _dup2... y devuelve log."""
    cols = pd.Series(df.columns)
    log = []
    if cols.duplicated().any():
        counts = {}
        new_cols = []
        for c in cols:
            if c in counts:
                counts[c] += 1
                new_name = f"{c}_dup{counts[c]}"
                new_cols.append(new_name)
                log.append(f"🧱 Columna duplicada '{c}' renombrada a '{new_name}'.")
            else:
                counts[c] = 0
                new_cols.append(c)
        df.columns = new_cols
    return df, log

def try_parse_dates(series: pd.Series):
    """Intenta convertir a fecha si es object. Devuelve (serie_convertida, ok_bool)."""
    if series.dtype != "object":
        return series, False
    try:
        s = pd.to_datetime(series, errors="raise", infer_datetime_format=True)
        return s, True
    except Exception:
        return series, False

def smart_to_numeric(series: pd.Series):
    """
    Convierte strings a numérico soportando formatos:
    - '1,234.56' (EN)
    - '1.234,56' (EU)
    - con espacios o '%'
    Devuelve (serie_convertida, ratio_numerico)
    """
    if series.dtype != "object" and not pd.api.types.is_categorical_dtype(series):
        return pd.to_numeric(series, errors="coerce"), series.notna().mean()

    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s+", "", regex=True).str.replace("%", "", regex=False)

    # Estrategia EN: separar miles con coma
    en = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    en_ratio = en.notna().mean()

    # Estrategia EU: separar miles con punto, decimal con coma
    eu = pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    eu_ratio = eu.notna().mean()

    if eu_ratio > en_ratio:
        return eu, eu_ratio
    else:
        return en, en_ratio

def classify_columns(df: pd.DataFrame, threshold=0.7):
    """
    1) Detecta fechas en columnas object.
    2) Convierte a numérico si > threshold de valores son numéricos.
    3) Resto -> categórica.
    Devuelve df convertido, listas de columnas y log.
    """
    log = []

    # 1) Detectar fechas solo en object
    for col in df.columns:
        if df[col].dtype == "object":
            conv, is_date = try_parse_dates(df[col])
            if is_date:
                df[col] = conv
                log.append(f"📅 '{col}' convertida a fecha (datetime).")

    # 2) Intentar numérico en objetos/categorías que no quedaron como fecha
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            conv, ratio = smart_to_numeric(df[col])
            if ratio >= threshold:
                df[col] = conv
                log.append(f"🔢 '{col}' convertida a numérica (ratio {ratio:.0%}).")
            else:
                df[col] = df[col].astype("category")
                log.append(f"🏷️ '{col}' clasificada como categórica (ratio numérico {ratio:.0%}).")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return df, numeric_cols, cat_cols, date_cols, log

def get_stock_index_col(df: pd.DataFrame):
    """Devuelve nombre real de columna 'Stock Index' sin importar mayúsculas/minúsculas; si no existe, None."""
    for c in df.columns:
        if c.strip().lower() == "stock index":
            return c
    return None

# ====== Carga de archivo ======
uploaded = st.file_uploader("📂 Sube tu archivo CSV", type=["csv"])
if not uploaded:
    st.info("Carga un CSV para iniciar el análisis.")
    st.stop()

# Leemos y deduplicamos nombres de columnas
df_raw = pd.read_csv(uploaded)
df, dup_log = deduplicate_columns(df_raw.copy())

# Clasificación/Conversión robusta
df, numeric_cols, cat_cols, date_cols, conv_log = classify_columns(df, threshold=0.7)

# ====== Vista previa y logs ======
st.subheader("👀 Vista previa")
st.dataframe(df.head())

with st.expander("🔎 Registro de conversiones automáticas"):
    for m in dup_log + conv_log:
        st.markdown(f"- {m}")

st.subheader("🗂️ Tipos detectados")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Numéricas**")
    st.write(numeric_cols if numeric_cols else "—")
with c2:
    st.markdown("**Categóricas**")
    st.write(cat_cols if cat_cols else "—")
with c3:
    st.markdown("**Fecha**")
    st.write(date_cols if date_cols else "—")

# ====== Calidad de datos ======
st.subheader("🧹 Calidad de datos")
nulls = df.isnull().sum().rename("Nulos")
# Outliers por IQR (solo numéricas)
outliers = {}
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
quality = pd.concat([nulls, pd.Series(outliers, name="Atípicos")], axis=1)
st.dataframe(quality)

# ====== Estadísticas descriptivas ======
if numeric_cols:
    st.subheader("📈 Estadísticas descriptivas (numéricas)")
    st.dataframe(df[numeric_cols].describe().T)

# ====== Boxplots (opción normalizar) ======
if numeric_cols:
    st.subheader("📦 Boxplots de variables numéricas")
    norm_box = st.checkbox("Normalizar con MinMaxScaler (0-1) antes del boxplot", value=False, key="box_norm")
    plot_df = df[numeric_cols].copy()
    if norm_box:
        scaler = MinMaxScaler()
        plot_df[numeric_cols] = scaler.fit_transform(plot_df[numeric_cols])

    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=plot_df[col], ax=axes[i])
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    st.pyplot(fig)

# ====== Histogramas categóricos ======
if cat_cols:
    st.subheader("📊 Frecuencias de variables categóricas")
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        df[col].value_counts(dropna=False).plot(kind="bar", ax=ax)
        ax.set_title(f"Frecuencia de {col}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# ====== Heatmap de correlación ======
if len(numeric_cols) >= 2:
    st.subheader("🔥 Matriz de correlación (verde = +, rojo = -)")
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "Correlación"})
    ax.set_title("Correlaciones entre variables numéricas")
    st.pyplot(fig)

# ====== Correlación entre dos variables (con opción MinMax) ======
# --- Análisis de correlación entre dos variables ---
st.write("### 🔗 Análisis de correlación entre dos variables numéricas")

if len(numeric_cols) >= 2:
    # Por defecto seleccionamos dos variables diferentes
    var1 = st.selectbox("Selecciona la primera variable", numeric_cols, index=0)
    var2 = st.selectbox("Selecciona la segunda variable", numeric_cols, index=1)

    # Checkbox para normalización con MinMaxScaler
    normalize_corr = st.checkbox("Normalizar variables con MinMaxScaler (0-1)", key="corr_norm")

    # Validamos que no sean la misma variable
    if var1 == var2:
        st.error("⚠️ No tiene sentido calcular la correlación de una variable consigo misma. Selecciona dos variables diferentes.")
    else:
        data_corr = df[[var1, var2]].dropna()

        if normalize_corr:
            scaler = MinMaxScaler()
            data_corr[[var1, var2]] = scaler.fit_transform(data_corr[[var1, var2]])

        # Convertimos en series 1D
        x = data_corr[var1].astype(float).squeeze()
        y = data_corr[var2].astype(float).squeeze()

        # Correlación de Pearson
        corr_value = x.corr(y, method="pearson")

        st.write(f"Coeficiente de correlación de **Pearson** entre **{var1}** y **{var2}**: `{corr_value:.2f}`")

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(x=x, y=y, ax=ax)
        ax.set_title(f"Correlación {var1} vs {var2}")
        st.pyplot(fig)




# ===




# ====== Tendencias temporales ======
if date_cols and numeric_cols:
    st.subheader("📈 Análisis de tendencia en el tiempo")
    date_col = st.selectbox("Columna de fecha", date_cols, key="trend_date")
    trend_var = st.selectbox("Variable numérica", numeric_cols, key="trend_var")

    period_dict = {"Diario": "D", "Semanal": "W", "Mensual": "M", "Trimestral": "Q", "Anual": "Y"}
    period_label = st.selectbox("Periodo de resumen", list(period_dict.keys()), index=2)
    period = period_dict[period_label]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    trend = (
        df.groupby(pd.Grouper(key=date_col, freq=period))[trend_var]
        .mean()
        .reset_index()
        .dropna(subset=[trend_var])
    )

    if trend.empty:
        st.warning("No hay datos para graficar en el periodo seleccionado.")
    else:
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.lineplot(x=trend[date_col], y=trend[trend_var], ax=ax, marker="o")
        ax.set_title(f"Tendencia de {trend_var} ({period_label})")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(f"Promedio de {trend_var}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# ====== Pivot Table por Stock Index ======
stock_col = get_stock_index_col(df)
if stock_col and date_cols and numeric_cols:
    st.subheader("📊 Pivot Table: promedio por fecha y Stock Index")
    date_for_pivot = st.selectbox("Fecha para pivot", date_cols, key="pivot_date")
    value_var = st.selectbox("Variable numérica a promediar", numeric_cols, key="pivot_value")

    pivot_df = pd.pivot_table(
        df,
        values=value_var,
        index=date_for_pivot,
        columns=stock_col,
        aggfunc="mean"
    )
    st.dataframe(pivot_df.head())

    fig, ax = plt.subplots(figsize=(11, 4))
    pivot_df.plot(ax=ax)
    ax.set_title(f"Tendencia promedio de {value_var} por {stock_col}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"Promedio de {value_var}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
