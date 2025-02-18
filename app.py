from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import random
import json
import numpy as np
import pickle
import requests
import json
import os
import re


# get value from enviroment variable
tenorflow_url = os.environ.get(
    'TENSORFLOW_URL', 'http://localhost:8501/v1/models/multilabel_model:predict')

def get_responce_from_model_server(msg):
    data = json.dumps(
        {"signature_name": "serving_default", "instances": [msg]})
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        tenorflow_url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions



# function to clean the word of any punctuation or special characters and lowwer it


def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.lower()
    return cleaned
    
#Defining a function to get genres name from model predicted value

def sentiment_predictor(prediction):
    
    temp=""
    
    if prediction > 0.5:
       temp = "Positive"
    else:
       temp = "Negative"

    return temp

def chatbot_response(msg):
    msg = cleanPunc(msg)
    pred = get_responce_from_model_server(msg)
    print(pred)
    pred = sentiment_predictor(pred[0][0])
    return pred


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    run_with_ngrok(app)
    app.run()
