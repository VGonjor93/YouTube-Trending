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

#Explore data
pp.ProfileReport(US_df) #Run this only on Jupyter

#Changing data types of date variables
US_df["publish_time"]=pd.to_datetime(US_df["publish_time"])
US_df["trending_date"]=pd.to_datetime(US_df["trending_date"], format="%y.%d.%m")
US_df["trending_date"]


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