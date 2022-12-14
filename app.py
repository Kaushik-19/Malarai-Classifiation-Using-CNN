import streamlit as st
import tensorflow as tf
from skimage import transform


# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model('model.h5', compile=False)
#     return model
  #model=tf.keras.models.load_model('model.h5')
#return model
@st.experimental_singleton
def load_model():
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/snehankekre/Kaushik-19/raw/main/Breccia_Rock_Classifier.h5', 'model.h5')
    return tensorflow.keras.models.load_model('model.h5')
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Malaria Detection Using CNN
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        np_image = transform.resize(image, (224, 224, 3))
        np_image = np.expand_dims(np_image, axis=0)
        img = np_image

        prediction = model.predict(np_image)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Infected', 'Healthy']
    string = "This Person is : "+class_names[np.argmax(predictions)]
    st.success(string)

