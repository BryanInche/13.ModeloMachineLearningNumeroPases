import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar el modelo guardado
#model_xgb = joblib.load('modelo_xgb.pkl') #Model 1: 6 variables
model_xgb = joblib.load('modelo_xgb_vf.pkl') #Model 2 : 3 variables

# Leer la base de datos
@st.cache_data
def load_data():
    # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo CSV
    #data = pd.read_csv('datatest_deploy_model1.csv')
    data = pd.read_csv('datatest_deploy_model2.csv')
    return data

data = load_data()

# Diseño de la aplicación Streamlit
st.title("Predicciones en tiempo real con XGBoost")

# Mostrar la base de datos cargada
st.write("Base de datos:")
st.write(data)

# Botón para realizar predicciones en toda la base de datos
if st.button('Realizar Predicciones en Toda la Base de Datos'):
    # Obtener las características de la base de datos
    #features = data[['capacidad_en_volumen_equipo_carguio_m3', 'capacidad_en_peso_equipo_carguio', 'capacidad_en_peso_equipo_acarreo',
    #                'tonelaje_camion_antes_cargaestabilizada', 'angulo_giro_promedio_pases', 'densidad_inicial_poligono_creado_tn/m3']]
    
    features = data[['capacidad_en_volumen_equipo_carguio_m3','angulo_giro_promedio_pases', 'densidad_inicial_poligono_creado_tn/m3']]

    # Realizar predicciones utilizando el modelo cargado
    predictions = model_xgb.predict(features)
    rounded_predictions = np.round(predictions).astype('int64')

    # Agregar las predicciones a la base de datos
    data['Prediccion'] = rounded_predictions

    # Mostrar la base de datos con las predicciones
    st.write("Base de datos con Predicciones:")
    st.write(data)
