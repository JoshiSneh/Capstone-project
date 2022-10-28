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

predict_threshold = os.environ.get(
    'pred_threshold', "0.2")

predict_threshold = float(predict_threshold)
# Get responce from tensorflow model server


def get_responce_from_model_server(msg):
    data = json.dumps(
        {"signature_name": "serving_default", "instances": [msg]})
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        tenorflow_url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions



# def genre_predictor(prediction):
#   prediction_dict = {}
 
#   #creating a dictionory of prediction and class name
#   for i,p in enumerate(prediction[0]):
#     prediction_dict[genres_name[i]] = p

#   genres = []

#   #Filtering the dictionary to get only the intents that are above the threshold
#   for key,value in prediction_dict.items():

#     if value > 0.2:
#       genres.append(key)

#   #Converting those filtered dictionary keys to text
#   text = 'predicted genres are: '

#   for i in genres:
#     text += i + ","

#   return text

# function to clean the word of any punctuation or special characters and lowwer it


def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.lower()
    return cleaned

def sentiment_predictor(prediction):
  
    temp=""
    
    if prediction >= 0.5:
       temp = "Positive"
    else:
       temp = "Negative"

    return temp

def chatbot_response(msg):
    msg = cleanPunc(msg)
    pred = get_responce_from_model_server(msg)
    pred = sentiment_predictor(pred)
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
