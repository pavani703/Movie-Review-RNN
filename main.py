import streamlit as st
import tensorflow as tf
from tensorflow import keras
import json

# Initialize session state to store the model
if 'model' not in st.session_state:
    st.session_state.model = None

def load_model_function():
    try:
        # Define the model architecture manually
        model = keras.Sequential([
            keras.layers.Embedding(10000, 32),
            keras.layers.SimpleRNN(32),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Load weights only
        model.load_weights('simple_rnn_imdb.h5')
        
        # Store in session state
        st.session_state.model = model
        st.success("Model loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the model file exists in the correct location")

# Add a button to load the model
st.button("Load Model", on_click=load_model_function)

# When making predictions, check if model is loaded
if st.button("Predict"):
    if st.session_state.model is None:
        st.warning("Please load the model first by clicking the 'Load Model' button")
    else:
        try:
            prediction = st.session_state.model.predict(preprocessed_input)
            # Rest of your prediction code
        except Exception as e:
            st.error(f"Prediction error: {e}")
