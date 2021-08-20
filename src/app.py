import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import uvicorn

def preProcess_data(text):
    text = text.lower()
    newText = re.sub('[^a-zA-Z0-9\s]','',text)
    newText = re.sub('rt','',newText)
    return newText

data = pd.read_csv('C:/Users/baoth/Desktop/Learn_Linux/deploy/sentiment/data/Sentiment.csv')
tokenizer = Tokenizer(num_words=2000,split=' ')
tokenizer.fit_on_texts(data['text'].values)

def my_pipeline(text):
    text_new = preProcess_data(text)
    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    X = pad_sequences(X,maxlen=28)
    return X

app = FastAPI()

@app.get('/')
def welcome_view():
    return {
        "WELCOME": "GO TO /docs route, or /post or send post request to /predict "
    }

@app.get('/predict',response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Text Emotion to be tested" />
        <input type="submit" />'''


@app.post('/predict')
def predict(text:str = Form(...)):
    clean_text = my_pipeline(text)
    load_model = tf.keras.models.load_model('sentiment.h5')
    prediction = load_model.predict(clean_text)
    sentiment = int(np.argmax(prediction))
    probability = max(prediction.tolist()[0])
    if sentiment == 0:
        t_sentiment = 'negative'
    elif sentiment == 1:
        t_sentiment = 'neutral'
    elif sentiment == 2:
        t_sentiment = 'positive'
    
    return {
        "ACTUAL SENTIMENT": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "Probability": probability
    }

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8080)