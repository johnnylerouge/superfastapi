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

test=FastAPI()

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

@test.get("/")
def read_root():
    return {"Hello": "World"}