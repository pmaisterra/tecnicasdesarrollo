# Import
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pickle

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
    if st.button('Load Image to predict'):
      imageto=load_image()
    uploaded_file = st.file_uploader("Upload Model")
    if uploaded_file is not None:
      clf2 = pickle.loads(uploaded_file.read())
    if st.button('Image prediction'):
      prediction =clf2.predict(imageto)
      st.succes=prediction

if __name__ == '__main__':
    main()
