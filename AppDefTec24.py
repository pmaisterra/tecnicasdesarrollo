import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow import keras
import pathlib
from PIL import Image

# Cargar el modelo con pickle
with open('model.sav', 'rb') as f:
    modelo = pickle.load(f)

data_train=pathlib.Path('seg_train')
data_test=pathlib.Path('seg_test')
# Cargar imágenes
# en este caso no partimos los datos ya que validación y entrenamiento se encuentran ya separados
image_size = (180, 180)
batch_size = 128
data_train = tf.keras.utils.image_dataset_from_directory(
  data_train,
  image_size=image_size,
  batch_size=batch_size,
)
  
data_test = tf.keras.utils.image_dataset_from_directory(
  data_test ,
  image_size=image_size,
  batch_size=batch_size,
)


# Definir función para crear y entrenar el modelo

def crear_modelo(filtros, tamaño, pool_size, neuronas,train_generator,validation_generator,num_classes):
    
    # Crear modelo
    model = Sequential([
        tf.keras.layers.Conv2D(filtros, tamaño, activation='relu', input_shape=(180, 180, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.Conv2D(filtros, tamaño, activation='relu'),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.Conv2D(filtros, tamaño, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=pool_size),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(neuronas, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compilar modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Entrenar modelo
    history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
    return model, history

def train_page():
  st.title('Entrenamiento de red neuronal para clasificar imágenes')
  # Cargar imágenes
  # Pedir al usuario que introduzca los parámetros de la red neuronal
  num_filters = st.slider('Número de filtros', 32, 256, 64, 32)
  filter_size = st.slider('Tamaño de filtro', 3, 7, 5, 2)
  pool_size = st.slider('Tamaño de pooling', 2, 5, 3, 1)
  num_neurons = st.slider('Número de neuronas', 64, 512, 128, 32)
  # Añadir un botón
  boton_hola = st.button('Haz click para entrenar el modelo')

  # Activar una acción al hacer clic en el botón
  if boton_hola:
    num_classes=len(data_train.class_names)
  # Crear y entrenar modelo
    if data_train is not None and data_test is not None:
      model, history = crear_modelo(num_filters, filter_size, pool_size, num_neurons,data_train,data_test,num_classes)
      # Mostrar gráficos de la precisión y la pérdida durante el entrenamiento
      st.write('Gráficos de precisión y pérdida durante el entrenamiento:')
      fig, ax = plt.subplots(1, 2, figsize=(10, 5))
      ax[0].plot(history.history['accuracy'], label='Precisión')
      ax[0].plot(history.history['val_accuracy'], label='Precisión en validación')
      ax[0].set_xlabel('Época')
      ax[0].set_ylabel('Precisión')
      ax[0].legend()
      ax[1].plot(history.history['loss'], label='Pérdida')
      ax[1].plot(history.history['val_loss'], label='Pérdida en validacion')

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
def validate_page():
 # Interfaz de Streamlit
 st.title('Clasificación de imágenes')

 # Cargar la imagen
 imagen = st.file_uploader('Selecciona una imagen', type=['jpg', 'jpeg', 'png'])

 imagen_path=imagen
 class_names=['buildings','forest','glacier','mountain','sea','street']
 if imagen is not None:
    # Mostrar la imagen en la interfaz
    imagen = Image.open(imagen)
    st.image(imagen, caption='Imagen cargada', use_column_width=True)
    
    # Clasificar la imagen y mostrar el resultado
    resultado = clasificar_imagen(modelo, imagen_path)
    clase_predicha=np.argmax(resultado)

    st.write(f'El modelo predice que la imagen es de la clase {class_names[clase_predicha]}')

# Creo un menú para entrenamiento y validación
menu = ["Entrenamiento", "Validación"]
select = st.sidebar.selectbox("Seleccione una opción", menu)

# Mostrar la página correspondiente a la opción seleccionada
if select == "Entrenamiento":
    train_page()
elif select == "Validación":
    validate_page()
