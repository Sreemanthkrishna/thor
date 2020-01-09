import os
import datetime
import hashlib
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
from database import list_users, verify, delete_user_from_db, add_user
from database import read_note_from_db, write_note_into_db, delete_note_from_db, match_user_id_with_note_id
from database import image_upload_record, list_images_for_user, match_user_id_with_image_uid, delete_image_from_db
from werkzeug.utils import secure_filename
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import math
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from tkinter import Tk
from tkinter.filedialog import askopenfilename


data=pd.read_csv(r'C:\Users\sreem\Desktop\samplelarge.csv')
data['text']=data.text.astype(str)
data.head()
data['length'] = data['text'].apply(len)
data.head()
data_classes = data[(data['stars']==1) | (data['stars']==2) | (data['stars']==3)| (data['stars']==4) | (data['stars']==5)]
data_classes.head()
print("class shape is",data_classes.shape)
# Seperate the dataset into X and Y for prediction
x = data_classes['text']
y = data_classes['stars']
print(x.head())
print(y.head())
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# CONVERTING THE WORDS INTO A VECTOR
global vocab
vocab = CountVectorizer(analyzer=text_process).fit(x)
print(len(vocab.vocabulary_))
r0 = x[0]
print(r0)
"""
vocab0 = vocab.transform([r0])
print(vocab0)
"""
#Now the words in the review number 78 have been converted into a vector.
#The data that we can see is the transformed words.
#If we now get the feature's name - we can get the word back!
"""
print("Getting the words back:")
print(vocab.get_feature_names()[19648])
print(vocab.get_feature_names()[10643])
print("Please Wait till the data is analysed")
#executed and working till here
"""
x = vocab.transform(x)
#Shape of the matrix:
print("Shape of the sparse matrix: ", x.shape)
#Non-zero occurences:
print("Non-Zero occurences: ",x.nnz)

# DENSITY OF THE MATRIX
density = (x.nnz/(x.shape[0]*x.shape[1]))*100
print("Density of the matrix = ",density)
# SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)
# Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB
global mnb
mnb = MultinomialNB()

mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test,predmnb))
print("Score:",round(accuracy_score(y_test,predmnb)*100,2))

print("Classification Report:/n",classification_report(y_test,predmnb))

pr = data['text'][278]
print(pr)
print("Actual Rating: ",data['stars'][278])
pr_t = vocab.transform([pr])

print("Predicted Rating:")
predicted=mnb.predict(pr_t)[0]
print(predicted)



# Dump the trained decision tree classifier with Pickle
mnbmodel = 'mnbmodel.pkl'
# Open the file to save as pkl file
mnbmodel = open(mnbmodel, 'wb')
pickle.dump(mnb, mnbmodel)
# Close the pickle instances
mnbmodel.close()


"""filename = 'mnbmodel.sav'
pickle.dump(mnb, open(filename, 'wb'))
#start Dataframe prdiction
"""
      