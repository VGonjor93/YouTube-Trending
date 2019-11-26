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
import pytz
import datetime


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


#Modeling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(US_df.drop(["tags", "channel_title", "title", "category_id", "video_id", "thumbnail_link", "description", "trending_date", "publish_time"], axis = 1), US_df["category_id"], test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train,y_train)
y_pred_rfc = random_forest_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_random_forest_classifier = confusion_matrix(y_test,y_pred_rfc)
print(cm_random_forest_classifier,end="\n\n")

numerator = cm_random_forest_classifier[0][0] + cm_random_forest_classifier[15][15]
denominator = sum(cm_random_forest_classifier[0]) + sum(cm_random_forest_classifier[15])
acc_svc = (numerator/denominator) * 100
print("Accuracy : ",round(acc_svc,2),"%")


from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train,y_train)
y_pred_xgb = xgb_classifier.predict(X_test)

cm_xgb_classifier = confusion_matrix(y_test,y_pred_xgb)
print(cm_xgb_classifier,end='\n\n')

numerator = cm_xgb_classifier[0][0] + cm_xgb_classifier[15][15]
denominator = sum(cm_xgb_classifier[0]) + sum(cm_xgb_classifier[15])
acc_xgb = (numerator/denominator) * 100
print("Accuracy : ",round(acc_xgb,2),"%")
#outliers

import seaborn as sns
sns.boxplot(x=US_df['like_dislike_ratio'])

from scipy import stats
z = np.abs(stats.zscore(US_df["likes"]))
print(z)
print(np.where(z > 20))


print(pd.Timedelta(US_df["trending_date"] - US_df["publish_time"]).days)



#Sentiment Analysis
#Title
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

vader = []
for title in US_df.title:
    vader.append(analyser.polarity_scores(title))

textblob = []

for title in US_df.title:
    blob = TextBlob(title)     
    textblob.append(blob.sentiment)
    
textblob[0:5]
vader[0:5]

Names = ['title_polarity', 'title_subjectivity']
df = pd.DataFrame.from_records(textblob, columns=Names)

US_df['title_polarity'] = df.title_polarity
US_df['title_subjectivity'] = df.title_subjectivity

US_df.head()

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
#Description
