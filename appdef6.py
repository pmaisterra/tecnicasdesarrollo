import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Cargar el modelo con pickle
with open('model.sav', 'rb') as f:
    modelo = pickle.load(f)

# Función para clasificar la imagen
def clasificar_imagen(modelo, imagenclass):
    # Mostrar la imagen original
    imagen_original = Image.open(imagenclass)
    st.image(imagen_original, caption='Imagen a predecir', use_column_width=True)

    # Reescalar la imagen
    tamano = (180, 180)
    image_scaled = imagen_original.resize(tamano)
    st.image(image_scaled, caption='Imagen reescalada', use_column_width=True)

    # Convertir la imagen a un array numpy
    imagen_array = np.array(image_scaled)
    
    #añadir el batch size 
    imagen_array = imagen_array[np.newaxis, ...]
    # Clasificar la imagen con el modelo
    resultado = modelo.predict(imagen_array)
    
    # Devolver el resultado
    return resultado

# Interfaz de Streamlit
st.title('Clasificación de imágenes')

# Cargar la imagen
imagen = st.file_uploader('Selecciona una imagen', type=['jpg', 'jpeg', 'png'])

imagen_path=imagen

if imagen is not None:
    # Mostrar la imagen en la interfaz
    imagen = Image.open(imagen)
    st.image(imagen, caption='Imagen cargada', use_column_width=True)
    
    # Clasificar la imagen y mostrar el resultado
    resultado = clasificar_imagen(modelo, imagen_path)
    st.write(f'El modelo predice que la imagen es de la clase {resultado}')