# Import
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
import pickle
from PIL import Image

# Título de la página

st.title('Clasificación de imágenes')
st.markdown("Introduce una imagen y comprueba como se clasifica")

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        image= cv2.resize(image_data,(224, 224))
        return image
def load_model():
    uploaded_file = st.file_uploader("Upload Model")
    if uploaded_file is not None:
      clf2 = pickle.loads(open(uploaded_file,'rb'))
      return clf2
def main():
    st.title('Image upload demo')
    uploaded_image = st.file_uploader("Upload Image")
    if uploaded_image is not None:
      img = uploaded_image.getvalue()
      st.image(img)
      imaged=Image.open(io.BytesIO(img))
      #images=np.asarray(img).astype('uint8')
      down_width = 180
      down_height = 180
      down_points = (down_width, down_height)
      imageto= imaged.resize((down_points))
    uploaded_file = st.file_uploader("Upload Model")
    if uploaded_file is not None:
      clf2 = pickle.loads(uploaded_file.read())
    if st.button('Image prediction'):
      prediction =clf2.predict(imageto)
      st.succes=prediction

if __name__ == '__main__':
    main()
