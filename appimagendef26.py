# Import
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
import pickle
from keras.preprocessing.image import load_img
# Título de la página

st.title('Clasificación de imágenes')
st.markdown("Introduce una imagen y comprueba como se clasifica")

def main():
    st.title('Image')
    uploaded_image = st.file_uploader("Upload Image")
    if uploaded_image is not None:
      img = uploaded_image.getvalue()
      st.image(img)
      # load the image
      img = load_img(img)
      images=np.asarray(img)
      down_width = 180
      down_height = 180
      down_points = (down_width, down_height)
      imageto= cv2.resize(images,(down_points))
      imageto=imaget0/255.0
    uploaded_file = st.file_uploader("Upload Model")
    if uploaded_file is not None:
      clf2 = pickle.loads(uploaded_file.read())
    if st.button('Image prediction'):
      prediction =clf2.predict(imageto)
      st.succes=prediction

if __name__ == '__main__':
    main()
