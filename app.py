import streamlit as st
import pandas as pd
import numpy as np
from predict import predict

st.title('Classifying Iris Flowers')
st.markdown("Toy model to play classify iris flower into \
    (setosa, versicolor, virginica) based on their sepal/petal \
    and length/width")

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sepal Characteristics")
    sepal_l = st.slider('Sepal Length (cm)', 0.0, 10.0, 5.0)
    sepal_w = st.slider('Sepal Width (cm)', 0.0, 10.0, 5.0)

with col2:
    st.subheader("Petal Characteristics")
    petal_l = st.slider('Petal Length (cm)', 0.0, 10.0, 5.0)
    petal_w = st.slider('Petal Width (cm)', 0.0, 10.0, 5.0)

if st.button('Predict Iris Type'):
    input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    prediction = predict(input_data)
    prediction = prediction[0]
    
    st.header("Prediction Result")
    st.write(f"The iris flower is predicted to be: **{prediction}**")
    
    # Display the corresponding iris image
    image_path = f"images/{prediction.lower()}.jpg"
    st.image(image_path, caption=f'{prediction} Iris Flower', width=300)
