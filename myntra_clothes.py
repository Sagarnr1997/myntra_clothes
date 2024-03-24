# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:07:07 2024

@author: Manjunath CP
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense




data = pd.read_csv("myntra_dataset.csv")

data['filename'] = data['filename'].apply(lambda x: os.path.join('C:/Users/sagar.MANJUNATH/Myntra_cloth_images', x))
image_data = []
for filename in data["filename"]:
    img = load_img(filename, target_size=(224, 224))  # Assuming image size is 224x224
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    image_data.append(img_array)
# Preprocess labels
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['sleeve_length'] = label_encoder.fit_transform(data['sleeve_length'])

# Convert data to numpy arrays
X = np.array(image_data)
y_gender = np.array(data['gender'])
y_sleeve_length = np.array(data['sleeve_length'])


# Define and compile models
model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='softmax')  # Output for gender (2 classes: male, female)
])

model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='softmax')  # Output for gender (2 classes: male, female)
])

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model1.fit(X, y_gender, epochs=1)
model2.fit(X, y_sleeve_length, epochs=1)


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
    if st.button('Predict'):
        # Preprocess the image as per your model requirements
        # For now, let's convert it to a numpy array
        image_resized = image.resize((224, 224))
        image_np = np.array(image_resized)
        image_np = image_np / 255.0  # Normalize the pixel values to [0, 1]
        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

        # Make predictions
        gender, sleeve_type = predict(image_np, model1, model2)

        st.write('Gender Prediction:', gender)
        st.write('Sleeve Type Prediction:', sleeve_type)