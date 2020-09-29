#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:46:47 2020

@author: siddharthsmac
"""

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

messages = pd.read_csv('/users/siddharthsmac/downloads/smsspamcollection/SMSSpamCollection', 
                       sep = '\t', names = ['labels', 'message'])

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['labels'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
spam_detection_model = nb.fit(X_train, y_train)

y_pred = spam_detection_model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_pred, y_test)
score = accuracy_score(y_pred, y_test)

