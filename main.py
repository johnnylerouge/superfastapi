from email import message
import imp
from typing import Optional
from anyio import open_signal_receiver
from nbformat import read
import pandas
import numpy
import plotly
from fastapi import Body, FastAPI
from pydantic import BaseModel
from pyrsistent import freeze
import joblib, os
from scipy.sparse import hstack


from sklearn import preprocessing
from spacy import load

app=FastAPI()
preprocess = open('/home/jo/notebook/preprocess', 'rb')
preprocess = joblib.load(preprocess)
tfidf_X1 = open('/home/jo/notebook/tfidf_X1', 'rb')
tfidf_X1 = joblib.load(tfidf_X1)
tfidf_X2 = open('/home/jo/notebook/tfidf_X2', 'rb')
tfidf_X2 = joblib.load(tfidf_X2)
rfc = open('/home/jo/notebook/rfc', 'rb')
rfc = joblib.load(rfc)
binarizer = open('/home/jo/notebook/binarizer', 'rb')
binarizer = joblib.load(binarizer)


class Post(BaseModel):
    title : str
    content : str

@app.post('/predict')
def display_post(message : Post):
    unseen_data={'Title': preprocess(message.title), 'Body': preprocess(message.content)}
    unseen_data=pandas.DataFrame(data=unseen_data, index=[0])
    tfidf_X1=tfidf_X1.transform(unseen_data.Body)
    tfidf_X2=tfidf_X2.transform(unseen_data.Title)
    tfidf_unseen=hstack([tfidf_X1, tfidf_X2])
    y_pred=rfc.predict(tfidf_unseen)
    pred_list=binarizer.inverse_transform(y_pred)
    pred_list=list([x for x in pred_list for x in x])
    str1=''
    for w in pred_list:
        str1+=w
        tag=str1
    return{tag}

@app.get("/")
def read_root():
    return {"Hello": "World"}