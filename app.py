
import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load trained model
model = load_model('twitter_sentimental_analysis_model.h5')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Streamlit UI
st.title("Twitter Sentiment Analysis")

user_input = st.text_area("Enter your tweet here")

if st.button("Analyze Sentiment"):
    if user_input.strip():

        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])

        padded_sequence = pad_sequences(
            sequence,
            padding='post',
            maxlen=99
        )

        # Prediction
        prediction = model.predict(padded_sequence)

        sentiment = np.argmax(prediction, axis=1)[0]

        # Sentiment labels
        sentiment_labels = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        st.success(f"Sentiment: {sentiment_labels[sentiment]}")

    else:
        st.warning("Please enter a tweet!")
