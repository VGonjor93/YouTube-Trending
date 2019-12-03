# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:08:50 2019

@author: poni6
"""

#Import libraries
import pandas as pd
import numpy as np

#Get data
US_df= pd.read_csv("Data\\USvideos.csv")

#Description
US_df["description"] = US_df["description"].replace(np.nan, '', regex=True)
description = list(US_df["description"])

#Cleaning
import re
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_descriptions(description):
    description = [REPLACE_NO_SPACE.sub("", line.lower()) for line in description]
    description = [REPLACE_WITH_SPACE.sub(" ", line) for line in description]
    
    return description

description_clean = preprocess_descriptions(description)




#No Stop Words(NSW)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for description in corpus:
        removed_stop_words.append(
            ' '.join([word for word in description.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

description_clean_nsw = remove_stop_words(description_clean)


#Stemming
def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in description.split()]) for description in corpus]

description_clean_stem = get_stemmed_text(description_clean)

#Lemmatization
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in description.split()]) for description in corpus]

import nltk
nltk.download('wordnet')
description_clean_lem = get_lemmatized_text(description_clean)

#Vectorization
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(description_clean)
X = cv.transform(description_clean)


#Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 20450 else 0 for i in range(40901)]
X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))


final_model = LogisticRegression(C=1)
final_model.fit(X_train, y_train)
print ("Final Accuracy: %s" 
       % accuracy_score(y_val, final_model.predict(X_val)))

feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
    