import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences    

# Initialize tokenizer
tokenizer = Tokenizer(num_words=10000)

# Initialize the model
model = keras.Sequential([
    keras.layers.Embedding(10000, 16, input_length=500),
    keras.layers.SimpleRNN(32, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Load the weights
try:
    model.load_weights('simple_rnn_imdb.h5', by_name=True, skip_mismatch=True)
except Exception as e:
    st.error(f"Error loading model weights: {e}")

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    
    # Pad the sequence
    padded_sequence = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
    
    return padded_sequence

# Streamlit UI
st.title("Movie Review Sentiment Analysis")

# Text input
review_text = st.text_area("Enter your movie review:", height=100)

# **Auto-predict as soon as text is entered**
if review_text.strip():  # Only predict if there is text
    try:
        # Preprocess the input text
        input_sequence = preprocess_text(review_text)
        
        # Make prediction
        prediction = model.predict(input_sequence)
        
        # Interpret results
        sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
        confidence = float(prediction[0][0]) if prediction[0][0] >= 0.5 else float(1 - prediction[0][0])

        # Display results
        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")

        st.info(f"Confidence: {confidence:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
