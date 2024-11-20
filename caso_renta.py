import streamlit as st
import joblib
import numpy as np

# Cargar el modelo y los escaladores una vez al inicio
@st.cache_resource
def load_resources():
    try:
        scaler = joblib.load('scaler.joblib')
        model = joblib.load('model.joblib')
        scaler_y = joblib.load('scaler_y.joblib')
        return scaler, model, scaler_y
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None, None

scaler, model, scaler_y = load_resources()

# Verificar que los modelos se cargaron correctamente
if not all([scaler, model, scaler_y]):
    st.stop()

# Título de la aplicación
st.title("Predictor de Alquiler")
st.header("Predecir el precio de alquiler")
st.write("Ingrese las características del inmueble:")

# Frontend: Entradas del usuario con validación
try:
    metros_cuadrados = float(st.text_input("Metros cuadrados:", value="16"))
    if metros_cuadrados <= 0:
        st.error("Los metros cuadrados deben ser mayores que 0.")
    
    n_baños = float(st.text_input("Número de baños:", value="1"))
    if n_baños <= 0:
        st.error("El número de baños debe ser mayor que 0.")
    
    tiene_ascensor = st.selectbox('¿Tiene ascensor?', ("Sí", "No"))
    conversion = {"Sí": 1, "No": 0}
    tiene_ascensor = conversion[tiene_ascensor]
    
    n_habitaciones = int(st.text_input("Número de habitaciones:", value="1"))
    if n_habitaciones <= 0:
        st.error("El número de habitaciones debe ser mayor que 0.")
    
    tiene_estacionamiento = st.selectbox('¿Tiene estacionamiento?', ("Sí", "No"))
    tiene_estacionamiento = conversion[tiene_estacionamiento]
    
    # Backend: Preparación de los datos
    X_list = [metros_cuadrados, n_baños, tiene_ascensor, n_habitaciones, tiene_estacionamiento]
    X = np.array(X_list, dtype=np.float64).reshape(1, -1)

    # Botón para realizar predicción
    if st.button("Predecir"):
        try:
            # Escalar las características de entrada
            X_scaled = scaler.transform(X)
            
            # Hacer predicción
            predicciones = model.predict(X_scaled).reshape(1, -1)
            predicciones = scaler_y.inverse_transform(predicciones)
            y = np.round(predicciones[0], 2)
            
            # Mostrar resultados
            st.success(f"El precio estimado del alquiler es: Bs {y[0]}")
        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")

except ValueError:
    st.error("Por favor, introduce valores numéricos válidos.")
