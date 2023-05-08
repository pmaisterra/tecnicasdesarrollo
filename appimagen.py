# Import
import streamlit as st
import pickle

model = pickle.load(open('model.sav','rb'))
# Título de la página

st.title('Clasificación de imágenes')
st.markdown("Introduce una imagen y comprueba como se clasifica")

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        image= cv2.resize(image_data,(224, 224))
    return image
def main():
    st.title('Image upload demo')
    imageto=load_image()
    if st.button('Image prediction'):
      prediction =model.predict(imageto)
      st.succes=prediction

if __name__ == '__main__':
    main()
