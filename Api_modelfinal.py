import streamlit as st
import joblib
import numpy as np

# Cargar el modelo guardado
#model_xgb = joblib.load('modelo_xgb.pkl') # Modelo_Var2
model_xgb = joblib.load('modelo_xgb_vf.pkl')  # Modelo_Var4: 'capacidad_en_volumen_equipo_carguio_m3','angulo_giro_promedio_pases','densidad_inicial_poligono_creado_tn/m3'


# Diseño de la aplicación Streamlit
st.title("Predicciones en tiempo real con XGBoost")

# Input para las variables
var1 = st.number_input('capacidad_en_volumen_equipo_carguio_m3', value=0.0, step=0.1)
#var2 = st.number_input('capacidad_en_peso_equipo_carguio', value=0.0, step=0.1)
#var3 = st.number_input('capacidad_en_peso_equipo_acarreo', value=0.0, step=0.1)
#var4 = st.number_input('tonelaje_camion_antes_cargaestabilizada', value=0.0, step=0.1)
var5 = st.number_input('angulo_giro_promedio_pases', value=0.0, step=0.1)
var6 = st.number_input('densidad_inicial_poligono_creado_tn/m3', value=0.0, step=0.1)

# Botón para realizar la predicción
if st.button('Realizar Predicción'):
    # Crear un array con los valores de las variables
    #input_features = np.array([[var1, var2, var3, var4, var5, var6]])
    input_features = np.array([[var1, var5, var6]])

    # Realizar la predicción utilizando el modelo cargado
    prediction = model_xgb.predict(input_features)
    rounded_prediction = np.round(prediction).astype('int64')

    # Mostrar la predicción
    st.write(f'Predicción: {rounded_prediction[0]}')
