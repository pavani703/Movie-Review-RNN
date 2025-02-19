import streamlit as st
import tensorflow as tf
from keras.models import load_model
import h5py

# Initialize session state to store the model
if 'model' not in st.session_state:
    st.session_state.model = None

def load_model_function():
    try:
        # Set legacy mode for SimpleRNN
        config = tf.compat.v1.ConfigProto()
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        
        # Custom load function to remove problematic parameters
        with h5py.File('simple_rnn_imdb.h5', 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is not None:
                model_config = model_config.decode('utf-8')
                import json
                config_dict = json.loads(model_config)
                
                # Remove time_major parameter from SimpleRNN layers
                for layer in config_dict['config']['layers']:
                    if 'time_major' in layer['config']:
                        del layer['config']['time_major']
                
                # Recreate model from modified config
                model = tf.keras.models.model_from_json(json.dumps(config_dict))
                
                # Load weights
                model.load_weights('simple_rnn_imdb.h5')
                st.session_state.model = model
                st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Rest of your code remains the same
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
