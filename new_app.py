import streamlit as st
import pickle
import numpy as np
from PIL import Image
import joblib


# Load your pickle files
model1 = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')

# Function to make predictions using the loaded models
def predict(image, model1, model2):
    # Your prediction logic here
    # For demonstration, let's just return some dummy predictions
    prediction1 = model1.predict(image)
    prediction2 = model2.predict(image)

    gender = "male" if prediction1 == 1 else "female"
    sleeve_type = "half-sleeve" if prediction2 == 1 else "full_sleeve"
    
    return gender, sleeve_type

# Streamlit UI
st.title('Image Upload and Prediction')
st.write('Upload an image (jpg, jpeg, png)')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform prediction if user uploads an image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform prediction if user uploads an image
    if st.button('Predict'):
        if model1 is not None and model2 is not None:
            # Preprocess the image as per your model requirements
            # For now, let's convert it to a numpy array
            image_np = np.array(image)

            # Make predictions
            gender, sleeve_type = predict(image_np, model1, model2)

            st.write('Gender Prediction:', gender)
            st.write('Sleeve Type Prediction:', sleeve_type)
        else:
            st.write('Error loading one or more models. Please check your model files and try again.')
