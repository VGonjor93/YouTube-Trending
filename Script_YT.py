# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:22:49 2019

@author: Victor Gonzalez Jornet
"""
#Change directory
import os
os.chdir("C:\\Users\poni6\Desktop\Data Analysis\YouTube")
os.getcwd()


#Import libraries
import pandas as pd
import json
import numpy as np
import pandas_profiling as pp



#Get data
US_df= pd.read_csv("Data\\USvideos.csv")

with open('Data/US_category_id.json', 'r') as myfile:
    US_id=myfile.read()
US_id_p = json.loads(US_id)

#Extracting  id and title from the json 
for row in US_id_p['items']:
        print(row['id'] + " : " + row["snippet"]["title"])

def getList_id(dict): 
    list = [] 
    for row in dict["items"]: 
        list.append(row["id"]) 
          
    return list
def getList_title(dict): 
    list = [] 
    for row in dict["items"]: 
        list.append(row["snippet"]["title"]) 
          
    return list
category_id = getList_id(US_id_p)
category_id = list(map(int, category_id))
category_title = getList_title(US_id_p)
print(category_id)
print(category_title)
category = dict(zip(category_id, category_title))
print(category)


#Replace category id in df for title
US_df["category_id"]
US_df["category_id"]=US_df['category_id'].map(category).fillna(US_df['category_id'])




#Changing data types of date variables and adding new vars
US_df["publish_time"]=pd.to_datetime(US_df["publish_time"])
US_df["trending_date"]=pd.to_datetime(US_df["trending_date"], format="%y.%d.%m")
US_df["like_dislike_ratio"]=US_df["likes"]/(US_df["likes"] + US_df["dislikes"])
US_df["comments_per_view"]=US_df["comment_count"]/US_df["views"]
US_df["likes_dislikes_per_view"]=(US_df["likes"] + US_df["dislikes"])/US_df["views"]
US_df["likes_per_view"]=US_df["likes"]/US_df["views"]
US_df["dislikes_per_view"]=US_df["dislikes"]/US_df["views"]

US_df["publish_time"] = US_df["publish_time"].dt.tz_convert(None)
US_df['publish_to_trending'] = US_df['trending_date']-US_df['publish_time']
US_df['publish_to_trending'].plot

#Explore data
pp.ProfileReport(US_df) #Run this only on Jupyter

US_df["category_id"].value_counts().plot("pie")

import seaborn as sns
sns.distplot(np.log(US_df['likes']+1))
sns.distplot(np.log(US_df['dislikes']+1))
sns.distplot(np.log(US_df['comment_count']+1))
sns.distplot(np.log(US_df['views']+1))

sns.distplot(np.log(US_df['like_dislike_ratio']+1))
sns.distplot(np.log(US_df['comments_per_view']+1))
sns.distplot(np.log(US_df['likes_dislikes_per_view']+1))
sns.distplot(np.log(US_df['likes_per_view']+1))
sns.distplot(np.log(US_df['dislikes_per_view']+1))

#Checking for duplicated rows
duplicateRowsDF = US_df[US_df.duplicated()]
print(duplicateRowsDF)
US_df=US_df.drop_duplicates()

US_df['category_id'].value_counts().plot("bar")
US_df['channel_title'].value_counts().plot("bar")


#outliers

import seaborn as sns
sns.boxplot(x=US_df['like_dislike_ratio'])

from scipy import stats
z = np.abs(stats.zscore(US_df["likes"]))
print(z)
print(np.where(z > 20))


#Sentiment Analysis
#Title
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

vader = []
for title in US_df.title:
    vader_score = analyser.polarity_scores(title)
    vader.append(vader_score)
        
textblob_polarity = []
for title in US_df.title:
    textblob_score = TextBlob(title)
    textblob_polarity.append(textblob_score.sentiment.polarity)

textblob_subjectivity = []
for title in US_df.title:
    textblob_score = TextBlob(title)
    textblob_subjectivity.append(textblob_score.sentiment.subjectivity)

US_df["title_polarity"] = textblob_polarity
US_df["title_subjectivity"] = textblob_subjectivity


import matplotlib.pyplot as plt
plot_data = US_df[US_df.title_polarity != 0]

US_df.plot(kind = "scatter", x = "title_polarity", y = "views")

fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram of the polarity values
plot_data.title_polarity.hist(bins=[-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
                                        -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 
                                        0.8, 0.9, 1],
             ax=ax,
             color="purple")

plt.title("Polarity UPR tweet")

plt.show()

    
#CLUSTERING
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

US_df["description"] = US_df["description"].replace(np.nan, '', regex=True)
description = list(US_df["description"])

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

description = remove_stop_words(description)

#Lemmatization
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in description.split()]) for description in corpus]

import nltk
nltk.download('wordnet')
description = get_lemmatized_text(description)

#TF-IDF
vec = TfidfVectorizer(stop_words="english")
vec.fit(description)
features = vec.transform(description)

US_df['category_id'].nunique()
cls = MiniBatchKMeans(n_clusters=16, random_state=0)
cls.fit(features)
cls.predict(features)

from sklearn.metrics import homogeneity_score
homogeneity_score(US_df.category_id, cls.predict(features))
from sklearn.metrics import completeness_score
completeness_score(US_df.category_id, cls.predict(features))

# reduce the features to 2D
ipca = IncrementalPCA(n_components=2, batch_size = 100)
reduced_features = ipca.fit_transform(features.toarray())
# reduce the cluster centers to 2D
reduced_cluster_centers = ipca.transform(cls.cluster_centers_)

plt.scatter(features[:,0], features[:,1], c=cls.predict(features))
plt.scatter(cls.cluster_centers_[:, 0], cls.cluster_centers_[:,1], marker='x', s=150, c='b')


#CLUSTERING TITLE

#TF-IDF
vec = TfidfVectorizer(stop_words="english")
vec.fit(US_df["title"])
features = vec.transform(US_df["title"])

US_df['category_id'].nunique()
cls = MiniBatchKMeans(n_clusters=16, random_state=0)
cls.fit(features)
cls.predict(features)

from sklearn.metrics import homogeneity_score
homogeneity_score(US_df.category_id, cls.predict(features))
from sklearn.metrics import completeness_score
completeness_score(US_df.category_id, cls.predict(features))


#BAG OF WORDS
US_df["description"] = US_df["description"].replace(np.nan, '', regex=True)
description = list(US_df["description"])

from nltk.tokenize import word_tokenize
import string

tokenized_sents = [word_tokenize(i) for i in description]
tokenized_sents = [[word.lower() for word in text.split()] for text in description]

def remove_punctuation(from_text):
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in from_text]
    return stripped

tokenized_sents = [remove_punctuation(i) for i in tokenized_sents]

from nltk.corpus import stopwords

tokenized_sents = [remove_stop_words(i) for i in tokenized_sents]

wordfreq = {}
for sentence in tokenized_sents:
    for token in sentence:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

key_to_delete = max(wordfreq, key=lambda k: wordfreq[k])
del wordfreq[key_to_delete]

def BOW(from_text):
    tokenized_sents = [word_tokenize(i) for i in description]
    tokenized_sents = [[word.lower() for word in text.split()] for text in description]
    tokenized_sents = [remove_punctuation(i) for i in tokenized_sents]
    tokenized_sents = [remove_stop_words(i) for i in tokenized_sents]
    wordfreq = {}
    for sentence in tokenized_sents:
        for token in sentence:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    return wordfreq
    