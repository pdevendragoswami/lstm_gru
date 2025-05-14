import streamlit as st
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


with open("tokenizer.pkl","rb")  as file:
    tokenizer = pickle.load(file)

model = load_model("lstm_grp.keras")

max_sequence_len = model.input_shape[1]+1

def predict_next_word(model,tokenizer,sentence,max_sequence_len):
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    if len(token_list)>=max_sequence_len:
        token_list = token_list[-(max_sequence_len-1)]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len,padding="pre")
    predicted  = model.predict(token_list)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None 



st.title("next word prediction with LSTM")

sentence = st.text_input("Enter the sequence of word")
if st.button("Predict Next word"):
    next_word = predict_next_word(model, tokenizer, sentence, max_sequence_len)
    st.write(f"the next word could be : {next_word}")