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
    tokenized_sents = [word_tokenize(i) for i in from_text]
    tokenized_sents = [[word.lower() for word in text.split()] for text in from_text]
    tokenized_sents = [remove_punctuation(i) for i in tokenized_sents]
    tokenized_sents = [remove_stop_words(i) for i in tokenized_sents]
    wordfreq = {}
    for sentence in tokenized_sents:
        for token in sentence:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    del wordfreq['']
    return wordfreq

US_BOW = BOW(description)


df2 = US_df['description'].groupby(US_df['category_id']).unique().apply(pd.Series)
df2 = df2.replace(np.nan, '', regex=True)

cat_list = df2.values.tolist()
desc_1 = cat_list[0]
desc_2 = cat_list[1]
desc_10 = cat_list[2]
desc_15 = cat_list[3]
desc_17 = cat_list[4]
desc_19 = cat_list[5]
desc_20 = cat_list[6]
desc_22 = cat_list[7]
desc_23 = cat_list[8]
desc_24 = cat_list[9]
desc_25 = cat_list[10]
desc_26 = cat_list[11]
desc_27 = cat_list[12]
desc_28 = cat_list[13]
desc_29 = cat_list[14]
desc_43 = cat_list[15]


BOW_1 = BOW(desc_1)
BOW_2 = BOW(desc_2)
BOW_10 = BOW(desc_10)
BOW_15 = BOW(desc_15)
BOW_17 = BOW(desc_17)
BOW_19 = BOW(desc_19)
BOW_20 = BOW(desc_20)
BOW_22 = BOW(desc_22)
BOW_23 = BOW(desc_23)
BOW_24 = BOW(desc_24)
BOW_25 = BOW(desc_25)
BOW_26 = BOW(desc_26)
BOW_27 = BOW(desc_27)
BOW_28 = BOW(desc_28)
BOW_29 = BOW(desc_29)
BOW_43 = BOW(desc_43)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

wc = WordCloud(background_color="white",width=1000,height=1000, max_words=10,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(BOW_2)
plt.imshow(wc)

#K MEANS TEST
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score

US_df["description"] = US_df["description"].replace(np.nan, '', regex=True)
description = list(US_df["description"])


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
cls = MiniBatchKMeans(n_clusters=500, random_state=0)
cls.fit(features)
cls.predict(features)
homogeneity_score(US_df.category_id, cls.predict(features))
completeness_score(US_df.category_id, cls.predict(features))

#K-means = 16 ->  0.07 / 0.09
# ------------------------- #
#K-means = 1  -> -2.44 / 1.00
#K-means = 2  ->  0.01 / 0.04
#K-means = 3  ->  0.01 / 0.04
#K-means = 4  ->  0.04 / 0.09
#K-means = 5  ->  0.04 / 0.08
#K-means = 6  ->  0.05 / 0.11
#K-means = 7  ->  0.03 / 0.07
#K-means = 8  ->  0.07 / 0.12
#K-means = 9  ->  0.08 / 0.13
#K-means = 10 ->  0.07 / 0.13
#K-means = 11 ->  0.06 / 0.09
#K-means = 12 ->  0.07 / 0.13
#K-means = 13 ->  0.07 / 0.10
#K-means = 14 ->  0.07 / 0.09
#K-means = 15 ->  0.08 / 0.13
#K-means = 20 ->  0.08 / 0.14
#K-means = 30 ->  0.13 / 0.15
#K-means = 40 ->  0.13 / 0.15
#K-means = 45 ->  0.13 / 0.16
#K-means = 50 ->  0.05 / 0.30
#K-means = 100->  0.17 / 0.19
#K-means = 500->  0.38 / 0.27



#Preparing DF for Tableau

Tableau_df = US_df
Tableau_df.title = Tableau_df.title.apply(lambda x: x.replace(',',''))
Tableau_df.channel_title = Tableau_df.channel_title.apply(lambda x: x.replace(',',''))
Tableau_df.tags = Tableau_df.tags.apply(lambda x: x.replace(',',''))
Tableau_df.thumbnail_link = Tableau_df.thumbnail_link.apply(lambda x: x.replace(',',''))
Tableau_df.description = Tableau_df.description.apply(lambda x: x.replace(',',''))

Tableau_df.to_csv(r'Data\\Tableau_df.csv')
