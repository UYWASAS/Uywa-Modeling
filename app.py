import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Modelo Predictivo Avícola", layout="wide")

# ----------- ESTILO CORPORATIVO -----------
st.markdown(
    """
    <style>
    body, .stApp {
        background: linear-gradient(120deg, #f3f6fa 0%, #e3ecf7 100%) !important;
    }
    .stSidebar, .stSidebarContent, .stSidebar * {
        background: #19345c !important;
        color: #fff !important;
    }
    section.main, section.main * {
        color: #19345c !important;
        font-family: 'Montserrat', 'Arial', sans-serif !important;
    }
    section.main > div:first-child {
        background: #fff !important;
        border-radius: 18px !important;
        box-shadow: 0 6px 32px 0 rgba(32, 64, 128, 0.11), 0 2px 8px 0 rgba(32,64,128,0.04) !important;
        padding: 2.5rem 2rem 2rem 2rem !important;
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
        min-height: 70vh !important;
    }
    h1, h2, h3, h4, h5, h6, .stTitle, .stHeader, .stSubheader, .stMarkdown, .stText, .stCaption {
        color: #19345c !important;
    }
    label, .stNumberInput label, .stTextInput label, .stSelectbox label, .stMultiSelect label, .stCheckbox label, .stRadio label {
        color: #19345c !important;
        font-weight: 600 !important;
    }
    .stNumberInput input, .stTextInput input, .stSelectbox, .stMultiSelect {
        background: #f4f8fa !important;
        border-radius: 6px !important;
        color: #19345c !important;
    }
    .stButton>button {
        background-color: #204080 !important;
        color: #fff !important;
        border-radius: 6px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("nombre_archivo_logo.png", width=90)
    st.markdown(
        """
        <div style='text-align: center;'>
            <div style='font-size:24px;font-family:Montserrat,Arial;color:#fff; margin-top: 10px;letter-spacing:1px;'>
                <b>UYWA-NUTRITION<sup>®</sup></b>
            </div>
            <div style='font-size:13px;color:#fff; margin-top: 5px; font-family:Montserrat,Arial;'>
                Nutrición de Precisión Basada en Evidencia
            </div>
            <hr style='border-top:1px solid #2e4771; margin: 10px 0;'>
            <div style='font-size:12px;color:#fff; margin-top: 8px;'>
                <b>Contacto:</b> uywasas@gmail.com<br>
                Derechos reservados © 2025
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("Modelo Predictivo de Resultados Productivos según Composición Nutricional de la Dieta")
st.write(
    "Sube tu archivo histórico (csv/xlsx) con los nutrientes analíticos y resultados productivos. "
    "Elige la variable productiva a predecir y entrena el modelo. Luego, modifica los nutrientes "
    "para predecir el resultado esperado según la dieta propuesta."
)

@st.cache_data
def cargar_archivo(file):
    if file is None:
        return None
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, sheet_name=0)

nutrientes_modelo = [
    "energia", "proteina", "lisina", "metionina", "treonina", "triptófano", "calcio",
    "fosforo", "sodio", "potasio", "cloro", "energia_proteina"
]
productivos_modelo = [
    "peso_final", "fcr", "gdp", "mortalidad", "consumo", "iep"
]

archivo = st.file_uploader("Carga tu archivo histórico (csv/xlsx)", type=["csv","xlsx"], key="file_predictivo")
df_hist = None
if archivo:
    df_hist = cargar_archivo(archivo)
    st.write("Vista previa de tus datos históricos:")
    st.dataframe(df_hist.head(20), use_container_width=True)
    cols = df_hist.columns.str.lower()
    nutrientes_disponibles = [n for n in nutrientes_modelo if n in cols]
    productivos_disponibles = [p for p in productivos_modelo if p in cols]
    if len(nutrientes_disponibles) < 7:
        st.error(f"Tu archivo debe incluir al menos 7 de las siguientes columnas: {nutrientes_modelo}")
    elif len(productivos_disponibles) == 0:
        st.error(f"Tu archivo debe contener al menos una de estas variables productivas como resultado: {productivos_modelo}")
    else:
        variable_pred = st.selectbox("Elige la variable productiva a predecir:", productivos_disponibles)
        nutrientes_usar = st.multiselect("Selecciona los nutrientes para el modelo:", nutrientes_disponibles, default=nutrientes_disponibles)
        if st.button("Entrenar modelo predictivo"):
            df_hist = df_hist.copy()
            X = df_hist[nutrientes_usar].astype(float)
            y = df_hist[variable_pred].astype(float)
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            lr = LinearRegression()
            rf.fit(X_train, y_train)
            lr.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)
            pred_lr = lr.predict(X_test)
            mae_rf = mean_absolute_error(y_test, pred_rf)
            mae_lr = mean_absolute_error(y_test, pred_lr)
            r2_rf = r2_score(y_test, pred_rf)
            r2_lr = r2_score(y_test, pred_lr)
            if r2_rf > r2_lr:
                modelo = rf
                modelo_tipo = "Random Forest"
                r2 = r2_rf
                mae = mae_rf
            else:
                modelo = lr
                modelo_tipo = "Regresión Lineal"
                r2 = r2_lr
                mae = mae_lr
            st.session_state["modelo_predictivo"] = modelo
            st.session_state["scaler_predictivo"] = scaler
            st.session_state["nutrientes_usar"] = nutrientes_usar
            st.session_state["variable_pred"] = variable_pred
            st.success(f"¡Modelo entrenado exitosamente! ({modelo_tipo})")
            st.info(f"R² validación: {r2:.3f} | MAE: {mae:.4f}")

            if modelo_tipo == "Random Forest":
                importancias = modelo.feature_importances_
                importancia_df = pd.DataFrame({"Nutriente": nutrientes_usar, "Importancia": importancias})
                importancia_df = importancia_df.sort_values("Importancia", ascending=False)
                st.markdown("#### Importancia de cada nutriente para la predicción:")
                st.dataframe(importancia_df)
                fig_imp = go.Figure(go.Bar(x=importancia_df["Nutriente"], y=importancia_df["Importancia"]))
                fig_imp.update_layout(title="Importancia relativa de los nutrientes", xaxis_title="Nutriente", yaxis_title="Importancia")
                st.plotly_chart(fig_imp, use_container_width=True)

        if "modelo_predictivo" in st.session_state and st.session_state.get("variable_pred") == variable_pred:
            st.markdown("---")
            st.markdown("### Predice el resultado productivo para una dieta propuesta")
            inputs = {}
            for nutr in st.session_state["nutrientes_usar"]:
                hist_vals = df_hist[nutr]
                val_min = float(hist_vals.min())
                val_max = float(hist_vals.max())
                val_med = float(hist_vals.mean())
                val_input = st.number_input(f"{nutr.capitalize()}",
                                            min_value=round(val_min*0.8, 3),
                                            max_value=round(val_max*1.2, 3),
                                            value=round(val_med, 3),
                                            step=0.001)
                inputs[nutr] = val_input
            X_pred = pd.DataFrame([inputs])[st.session_state["nutrientes_usar"]]
            scaler = st.session_state["scaler_predictivo"]
            X_pred_scaled = scaler.transform(X_pred)
            modelo = st.session_state["modelo_predictivo"]
            pred = modelo.predict(X_pred_scaled)[0]
            st.success(f"Predicción para {st.session_state['variable_pred'].upper()}: **{pred:.4f}**")
            prom_hist = float(df_hist[st.session_state["variable_pred"]].mean())
            st.markdown(f"Promedio histórico: **{prom_hist:.4f}**")
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                delta={'reference': prom_hist},
                title={"text": f"{st.session_state['variable_pred'].upper()}"},
                gauge={'axis': {'range': [min(prom_hist, pred)*0.9, max(prom_hist, pred)*1.1]}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Distribución histórica de la variable predicha:")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=df_hist[st.session_state["variable_pred"]], nbinsx=20, name="Histórico"))
            fig_hist.add_trace(go.Scatter(x=[pred], y=[0], mode='markers', marker=dict(size=15, color='red'), name="Predicción"))
            fig_hist.update_layout(title=f"Histórico de {st.session_state['variable_pred'].upper()} vs Predicción")
            st.plotly_chart(fig_hist, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:11px;color:#fff;text-align:center'>Desarrollado como MVP para <b>UYWA-NUTRITION®</b></div>",
    unsafe_allow_html=True
)
