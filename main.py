# -*- coding: utf-8 -*-
"""
\fail main.py
\brief Emotion classification model method
\details In this file we are going to create a method in which 2 csv files will be given as parameters (one for training, one for testing)
\authors Andreas Neocleous, Maria Katsama
\date 22/05/2022 
"""

import pandas as pd
import numpy as np
import neattext as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


def train_test(train_csv,test_csv):
    
    #Creating DataFrames from the paths of the file provided as arguments
    train  = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    
    #Cleaning our training data
    train['clean_text'] = train['text'].apply(nfx.remove_stopwords)
    train['clean_text'] = train['clean_text'].apply(nfx.remove_punctuations)
    train['clean_text'] = train['clean_text'].apply(nfx.remove_userhandles)
    train['clean_text'] = train['clean_text'].apply(nfx.remove_multiple_spaces)
    train['clean_text'] = train['clean_text'].apply(nfx.remove_emojis)
    Xfeatures = train['clean_text']
    y_train = train['emotion']
    cv = CountVectorizer()
    X = cv.fit_transform(Xfeatures)
    X.toarray()
    x_train = X
    
    #Build and train our model
    svm_model = svm.SVC(kernel='linear', break_ties=True)
    svm_model.fit(x_train, y_train)
    
    #Cleaning testing data
    test['clean_text'] = test['text'].apply(nfx.remove_stopwords)
    test['clean_text'] = test['clean_text'].apply(nfx.remove_punctuations)
    test['clean_text'] = test['clean_text'].apply(nfx.remove_userhandles)
    test['clean_text'] = test['clean_text'].apply(nfx.remove_multiple_spaces)
    test['clean_text'] = test['clean_text'].apply(nfx.remove_emojis)
    Xfeatures_test = test['clean_text']
    
    #Predicting testing DataFrame line-by-line and printing to .txt file
    with open('predictions.txt', 'w') as p:
        for i in range(0,len(Xfeatures_test)):
            sample_t = [str(Xfeatures_test[:][i])]
            vect = cv.transform(sample_t).toarray()
            prediction = svm_model.predict(vect)
            #print(prediction[0])
            p.write(prediction[0])
            p.write('\n')
    p.close()
    