#!/usr/bin/env python
# coding: utf-8

# In[49]:


from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[52]:


df_raw = pd.read_csv("cleaned_data_num.csv")
df = df_raw.set_index('id').drop(columns = 'breed')
Y = df.iloc[:, :18]
X = df.iloc[:, 18:37]


# In[53]:


# a series with cats' IDs to connect to prediction later
#cats_ids = df.index
#adopters_ids = pd.Series(df.index, index = df['id']).drop_duplicates()

#model_columns = joblib.load("model_columns.pkl")

#json_ = request.json
#query = pd.DataFrame(json_)
#query = query.reindex(columns=model_columns, fill_value=0)
#test prediction
#query = pd.DataFrame([[1,2,4,3,2,5,4,3,4,5,3,4,5,2,3,4,5,4,3]])
#query.columns = model_columns
            
#output = pd.DataFrame(prediction).sort_values(by = 0, ascending = False).iloc[:11].drop(columns = 0).reset_index().to_json()
#return jsonify({'output': output})


# In[54]:


# MODEL
#df = df_raw.drop(columns = 'breed')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=19)
lr = LinearRegression()
lr.fit(X_train,y_train)

joblib.dump(lr, 'model.pkl')
print("Model dumped!")
lr = joblib.load('model.pkl')

model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")


# In[ ]:




