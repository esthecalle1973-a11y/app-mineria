
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de Proceso de Flotación",
    page_icon="🧪",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'modelo_xgboost_final.joblib'.
model = load_model('mineria_model.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos .
    """)

    # Slider para el caudal de alimentación de amina
    AminaFlow = st.slider(
        label='Flujo de amina',
        min_value=241,
        max_value=739,
        value=300, # Valor inicial
        step=1
    )
    st.caption("Representa el flujo de amina.")

    # Slider para flujo e aire
    AirFlow = st.slider(
        label='Flujo de aire en la columna de flotación 01',
        min_value=175,
        max_value=372,
        value=180,
        step=1
    )
    st.caption("El flujo de aire inyectado.")

    # Slider para la diferencia de presión
    IronConcentrate = st.slider(
        label='Porcentaje de hierro en el concentrado',
        min_value=62,
        max_value=68,
        value=65,
        step=1
    )
    st.caption("Porcentaje de hierro en el concentrado.")

# --- Contenido de la Página Principal ---
st.title("🧪 Predictor de % de concentración de sílica en Prcoceso de Flotación")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir el porcentaje de sílice  en un proceso de flotación basándose en parámetros operativos claves. Los parámetros evaluados fueron: flujo de aire en columna de flotación 01, flujo de amina y porcentaje de hierro en el concentrado. 

**Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:**
- **Optimizar** las condiciones de operación como: flujo de aire, flujo de amina y porcentaje de hierro. 
- **Predecir** el impacto de las variables mencionadas anteriormente en el porcentaje de silica en el concentrado.
- **Solucionar** problemas potenciales simulando diferentes escenarios mediante un análisis multivariable.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir porcentaje de silica en el concentrado', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        df_input = pd.DataFrame({
            'VP1% Iron Concentrate': [IronConcentrate],
            'VP2Flotation Column 01 Air Flow': [AirFlow],
            'VP3Amina Flow': [AminaFlow]
        })

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Rendimiento Predicho:** `{prediction_value[0]:.2f}%`")
            st.info("Este valor representa el porcentaje de silica en el concentrado.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra porcentaje de silica contenida en el concentrado.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (XGBoost Optimizado)
    * **Propósito:** Predecir el porcentaje de silica en el concentrado.
    * **Características Usadas:** Flujo de aire en columna de flotación 01, flujo de amina y porcentaje de hierro en el concentrado.
    """)
