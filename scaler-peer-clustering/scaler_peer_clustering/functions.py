import numpy as np 
import pandas as pd 
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def basic_info(df):

    """
    Takes a DataFrame as input, and gives the basic info like shape, missing values count, duplicated rows, unique values and dtypes of the features
    
    Args:
        df (pandas.DataFrame): The DataFrame for which you want the details of

    Returns: 
        None 
    """
    print(f"shape of the date : \n\trows = {df.shape[0]}, columns = {df.shape[1]}\n")
    missing_val_count = df.isna().sum().sum()
    print(f"missing values: \n\tcount = {missing_val_count}")
    if missing_val_count != 0:
        missing_data = df.isna().sum().reset_index().rename({"index" : "feature", 0 : "missing_val_count"}, axis = 1)
        missing_data =  missing_data[missing_data.missing_val_count > 0]
        missing_data["missing_val_percentage"] = np.round((missing_data["missing_val_count"] / df.shape[0]) * 100, 2)
        missing_data = missing_data.sort_values(by = "missing_val_count", ascending = False)
        display(missing_data)

    print(f"duplicated records: \n\tcount = {df.duplicated().sum()}\n")
    print(f"Unique Values : ")
    nunique_vals = df.nunique().reset_index().rename({"index" : "feature", 0 : "nunique_vals"}, axis = 1)
    display(nunique_vals)

    display(df.dtypes.reset_index().rename(columns = {"index" : "fetaure", 0 : "data type"}))

    return missing_data


def preprocess_string(string):
    new_string= re.sub('[^A-Za-z ]+', '', string).lower().strip()
    return new_string

def segment(a,b_50,b_75):
    if a<b_50:
        return 3
    elif a>=b_50 and a<=b_75:
        return 2
    elif a>=b_75:
        return 1

def get_top(group, n=10, head = True):
    if head == True:
        return group.sort_values(by='ctc', ascending=False).head(n)
    return group.sort_values(by='ctc', ascending=False).tail(n)

def silhouette_method(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k+1):
        print("KMeans for k = {}, started.....".format(k))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        print("KMeans model builded for k = {}, calculating silhouette_score.....".format(k))
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)
        print("KMeans for k = {}, ended......\n".format(k))
    return silhouette_scores