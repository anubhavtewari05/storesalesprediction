#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error , mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


# ## Label Encoding
# 

# In[6]:


def label_encode(df_train):
    le = LabelEncoder()
    df_train['item_fat_content']=le.fit_transform(df_train['item_fat_content'])
    df_train['item_type']=le.fit_transform(df_train['item_type'])
    df_train['outlet_size']=le.fit_transform(df_train['outlet_size'])
    df_train['outlet_location_type']=le.fit_transform(df_train['outlet_location_type'])
    df_train['outlet_type']=le.fit_transform(df_train['outlet_type'])
    return le


# ## Standardization

# In[7]:


def standard_scaler(X_train,X_test):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return sc;


# In[ ]:




