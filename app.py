import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Configuración general ---
st.set_page_config(page_title="EDA Interactivo", layout="wide")

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3 { color: #333; }
    .stSelectbox label { font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Explorador de Datos (EDA) Interactivo")

# --- Carga de archivo ---
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Identificación de tipos de variables
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    # Intentar convertir columnas que parecen fechas
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                if col not in date_cols:
                    date_cols.append(col)
            except Exception:
                pass

    st.subheader("👀 Vista previa de los datos")
    st.write(df.head())

    # --- Resumen de datos ---
    st.subheader("📑 Resumen de datos")
    st.write("**Variables numéricas:**", numeric_cols)
    st.write("**Variables categóricas:**", categorical_cols)
    st.write("**Variables de fecha:**", date_cols)

    # Datos nulos
    st.subheader("📉 Datos nulos por columna")
    null_summary = df.isnull().sum()
    st.write(null_summary)

    # Resumen estadístico de numéricas
    st.subheader("📈 Estadísticas descriptivas")
    st.write(df.describe())

    # --- Outliers ---
    st.subheader("🚨 Detección de valores atípicos (Z-score > 3)")
    from scipy.stats import zscore
    outliers = {}
    for col in numeric_cols:
        if df[col].dtype in ["int64", "float64"]:
            z_scores = zscore(df[col].dropna())
            outliers[col] = (abs(z_scores) > 3).sum()
    st.write(outliers)

    # --- Histogramas de categóricas ---
    if categorical_cols:
        st.subheader("📊 Histogramas de variables categóricas")
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Frecuencia de {col}")
            st.pyplot(fig)

    # --- Boxplots de numéricas ---
    if numeric_cols:
        st.subheader("📦 Boxplots de variables numéricas")
        normalize = st.checkbox("Normalizar datos antes del boxplot (MinMaxScaler)")
        plot_data = df[numeric_cols].copy()

        if normalize:
            scaler = MinMaxScaler()
            plot_data[numeric_cols] = scaler.fit_transform(plot_data[numeric_cols])

        fig, axes = plt.subplots(nrows=len(numeric_cols) // 2 + len(numeric_cols) % 2, ncols=2, figsize=(12, 3 * (len(numeric_cols) // 2)))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(y=plot_data[col], ax=axes[i])
            axes[i].set_title(f"Boxplot de {col}")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig)

    # --- Heatmap de correlaciones ---
    if len(numeric_cols) > 1:
        st.subheader("🔥 Matriz de correlación")
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            cbar_kws={'label': 'Correlación'}, ax=ax
        )
        ax.set_title("Mapa de calor de correlaciones")
        st.pyplot(fig)

        # Análisis de correlación entre 2 variables
        st.subheader("🔗 Análisis de correlación entre dos variables")
        var1 = st.selectbox("Selecciona la primera variable", numeric_cols)
        var2 = st.selectbox("Selecciona la segunda variable", numeric_cols)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df[var1], y=df[var2], ax=ax)
        ax.set_title(f"Correlación entre {var1} y {var2}")
        st.pyplot(fig)

    # --- Gráficos de tendencia si hay fechas ---
    if date_cols and numeric_cols:
        st.subheader("📈 Gráficos de tendencia en el tiempo")
        date_col = date_cols[0]  # usar la primera columna fecha encontrada
        trend_var = st.selectbox("Selecciona variable numérica para graficar tendencia", numeric_cols)

        # Diccionario de periodos
        period_dict = {"Diario": "D", "Semanal": "W", "Mensual": "M", "Trimestral": "Q", "Anual": "Y"}
        period_label = st.selectbox("Selecciona el periodo de resumen", list(period_dict.keys()))
        period = period_dict[period_label]

        # Agrupación por periodo
        trend_data = df.groupby(pd.Grouper(key=date_col, freq=period))[trend_var].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=trend_data[date_col], y=trend_data[trend_var], ax=ax, marker="o")
        ax.set_title(f"Tendencia de {trend_var} ({period_label})")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Pivot Table ---
    if "stock index" in [c.lower() for c in df.columns] and date_cols and numeric_cols:
        st.subheader("📊 Pivot Table interactiva con Stock Index")
        stock_col = [c for c in df.columns if c.lower() == "stock index"][0]

        pivot_var = st.selectbox("Selecciona variable numérica para el pivot table", numeric_cols)

        pivot_table = pd.pivot_table(
            df,
            values=pivot_var,
            index=date_cols[0],
            columns=stock_col,
            aggfunc="mean"
        )

        st.write(pivot_table.head())

        fig, ax = plt.subplots(figsize=(10, 4))
        pivot_table.plot(ax=ax)
        ax.set_title(f"Tendencia promedio de {pivot_var} por {stock_col}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
