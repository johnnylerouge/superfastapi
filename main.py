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
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import sklearn
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from scipy.sparse import hstack
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import jaccard_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import preprocessing
from spacy import load
import re
from bs4 import BeautifulSoup
import string 

app=FastAPI()


df = pd.read_csv('/home/jo/Téléchargements/filteredresults.csv')

tfidf_X1=open('/home/jo/notebook/tfidf_X1', 'rb')
tfidf_X1=load(tfidf_X1)
tfidf_X2=open('/home/jo/notebook/tfidf_X2', 'rb')
tfidf_X2=load(tfidf_X2)
rfc=open('/home/jo/notebook/rfc', 'rb')
rfc=load(rfc)
binarizer=open('/home/jo/notebook/binarizer', 'rb')
binarizer=load(binarizer)



def preprocess(text):
    text = BeautifulSoup(text, 'lxml')
    text = text.get_text()
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)    
    text = text.strip(' ')
    punct = set(string.punctuation) 
    text = "".join([ch for ch in text if ch not in punct])
    stop_words = set(stopwords.words('english'))
    
    words=token.tokenize(text)
    
    text=[word for word in words if not word in stop_words]
    text=' '.join(map(str, text))
    text=token.tokenize(text)
    lemm_list=[]
    for word in text:
        x=lemmatizer.lemmatize(word, pos='v')
        lemm_list.append(x)
    text=' '.join(map(str, lemm_list))
    return text

def tags_normalization(text):
    text=text.replace('<','').replace('>', ' ')
    return text


df['Body']=df.Body.apply(filter_text)
df['Body']=df.Body.apply(filtering_stopwords)
df['Body']=df.Body.apply(lemmatize)
df['Title']=df.Title.apply(filter_text)
df['Title']=df.Title.apply(filtering_stopwords)
df['Title']=df.Title.apply(lemmatize)
df['Tags']=df.Tags.apply(tags_normalization)

df["Tags"] = df['Tags'].apply(lambda x:x.split())
flat_list = [word for w in df['Tags'].values for word in w]

keywords = nltk.FreqDist(flat_list)
keywords.most_common(30)

# retourne un dictionnaire où chaque mot est associé à sa fréquence d'apparition
keywords = nltk.FreqDist(flat_list)

keywords = nltk.FreqDist(keywords)

frequencies_words = keywords.most_common(100)
tags_features = [word[0] for word in frequencies_words]

def tags_filter(text):
    return list(set(text) & set(tags_features))

df['Tags']=df['Tags'].apply(tags_filter)

df = df[['Body', 'Tags', 'Title']]

class Item(BaseModel):
    content : str
    title : str



@app.get("/predict")
def tag_predict(x: Item.content, y: Item.title):
    unseen_data={'Title': preprocess(y), 'Body': preprocess(x)}
    unseen_data=pd.DataFrame(data=unseen_data, index=[0])
    tfidf_X1=tfidf_X1.transform(unseen_data.Body)
    tfidf_X2=tfidf_X2.transform(unseen_data.Title)
    tfidf_unseen=hstack([tfidf_Xa, tfidf_Xb])
    y_pred=rfc.predict(tfidf_unseen)
    pred_list=binarizer.inverse_transform(y_pred)
    return {"predicted tags": pred_list}




@app.get("/")
def read_root():
    return {"Hello": "World"}
