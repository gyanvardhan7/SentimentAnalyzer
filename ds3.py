
# coding: utf-8

# In[6]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
from sklearn.cross_validation import train_test_split

get_ipython().magic(u'matplotlib inline')


df = pd.read_csv("https://data.gov.in/sites/default/files/air_quality_PM10_2012.csv",header = None)

header = ["state","city","location","station code","Type","Category of ES","No. of mon. days","Min","Max","PM10","10 p","90 p","Std","Percentage exceeded","Air Quality"]
df.columns = header

df = df.drop(df.index[0])


df["Air Quality"] = df["Air Quality"].astype('category')
df["AQ"] = df["Air Quality"].cat.codes
df["PM10"] = pd.to_numeric(df["PM10"], errors='coerce')

df["Percentage exceeded"] = pd.to_numeric(df["Percentage exceeded"], errors='coerce')

mean = df["PM10"].mean()
df["PM10"].replace(np.nan, mean)

mean1 = df["Percentage exceeded"].mean()
df["Percentage exceeded"].replace(np.nan, mean1)

df[df.AQ != 4]


X = df['PM10']
Y = ['AQ']
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0)

