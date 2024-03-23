import streamlit as st
import pickle
import numpy as np
from PIL import Image

    

def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print("Error loading pickle file:", e)
        return None


# Function to make predictions using the loaded models
def predict(image, model1, model2):
    # Your prediction logic here
    # For demonstration, let's just return some dummy predictions
    prediction1 = model1.predict(image)
    prediction2 = model2.predict(image)

    gender = "male" if prediction1 == 1 else "female"
    sleeve_type = "half-sleeve" if prediction2 == 1 else "full_sleeve"
    
    return gender, sleeve_type

# Load your pickle files
model1 = load_model('model1.pkl')
model2 = load_model('model2.pkl')

# Streamlit UI
st.title('Image Upload and Prediction')
st.write('Upload an image (jpg, jpeg, png)')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform prediction if user uploads an image
    if st.button('Predict'):
        # Preprocess the image as per your model requirements
        # For now, let's convert it to a numpy array
        image_np = np.array(image)

        # Make predictions
        gender, sleeve_type = predict(image_np, model1, model2)

        st.write('Gender Prediction:', gender)
        st.write('Sleeve Type Prediction:', sleeve_type)
