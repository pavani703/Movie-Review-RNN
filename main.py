# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow import keras
from keras.models import load_model
import os

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
if os.path.exists('simple_rnn_imdb.h5'):
    print("Model file exists")
    try:
        model = load_model('simple_rnn_imdb.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("Model file not found")

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# Initialize session state to store the model
if 'model' not in st.session_state:
    st.session_state.model = None

def load_model_function():
    try:
        # Add custom_objects with legacy compatibility
        st.session_state.model = load_model('simple_rnn_imdb.h5', compile=False)
        st.success("Model loaded successfully!")
    except Exception as e:
        try:
            # If first attempt fails, try with legacy mode
            tf.keras.utils.disable_interactive_logging()
            with tf.keras.utils.custom_object_scope({'SimpleRNN': tf.keras.layers.SimpleRNN}):
                st.session_state.model = load_model('simple_rnn_imdb.h5', compile=False)
            st.success("Model loaded successfully in legacy mode!")
        except Exception as e2:
            st.error(f"Error loading model: {e2}")

# Add a button to load the model
st.button("Load Model", on_click=load_model_function)

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    if st.session_state.model is None:
        st.warning("Please load the model first by clicking the 'Load Model' button")
        sentiment = 'Error'
    else:
        try:
            prediction = st.session_state.model.predict(preprocessed_input)
            sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
        except Exception as e:
            st.error(f"Prediction error: {e}")
            sentiment = 'Error'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0] if st.session_state.model is not None else "Model not loaded"}')
else:
    st.write('Please enter a movie review.')

