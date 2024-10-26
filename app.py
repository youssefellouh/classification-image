import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Interface Streamlit
st.header('Image Classification Model')
model = load_model('Image_classify.keras')
data_cat = ['lemon', 'lettuce', 'mango', 'orange']
img_height = 180
img_width = 180

# Bouton pour ajouter une image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charge et prépare l'image
    image = Image.open(uploaded_file).convert('RGB')  # Conversion en RGB pour éviter les canaux Alpha
    image = image.resize((img_height, img_width))  # Redimensionne l'image
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)  # Ajout d'une dimension pour correspondre au modèle

    # Prédiction
    prediction = model.predict(img_bat)
    score = tf.nn.softmax(prediction)

    # Affichage des résultats
    st.image(image, width=200)
    st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
    st.write('With accuracy of ' + str(np.max(score) * 100) + '%')
else:
    st.write("Please upload an image file.")
