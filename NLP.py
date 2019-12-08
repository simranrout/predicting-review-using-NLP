# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:17:46 2019

@author: DELL
"""

import numpy as np
import pandas as pd
#from nltk.words import stopwords
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#print(dataset)
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#-------data preparation
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word  in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
#   --------------bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features= 1500)
X=cv.fit_transform(corpus).toarray()#it will create Sparcematrix for us which is nothing but the matrix of features and which will used to train the classification model
#print(S)
Y=dataset.iloc[:,1].values#storning depedent value or response
#print(Y)
#X is indepedent value and Y is depedent value 
#--------------------classification
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=0) #to train the ann on 8000 observation and test it on 2000 observation
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()#we have to create its object to call fit_transform
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#fitting naive bayes to training model
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)


#Creation of confusion matrix to know the accuracy..... (number of correct prediction)/total number of predication

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test,y_pred)
acc=(81+53)/200
print('accuracy is:-',acc)
