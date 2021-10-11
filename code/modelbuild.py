#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# ## Model Building using Linear Regression 

# In[16]:


def linear_regression(X_train_std,X_test_std,Y_train,Y_test):
    lr = LinearRegression()
    lr.fit(X_train_std,Y_train)
    Y_pred_lr =lr.predict(X_test_std)
    print("Linear Regression")
    print(r2_score(Y_test,Y_pred_lr))
    print(mean_absolute_error(Y_test,Y_pred_lr))
    print(np.sqrt(mean_squared_error(Y_test,Y_pred_lr)))
    return lr


# ## Model Building using Random Forest Regression

# In[17]:


def random_forest(X_train_std,X_test_std,Y_train,Y_test):
    rf = RandomForestRegressor()
    rf.fit(X_train_std,Y_train)
    Y_pred_rf = rf.predict(X_test_std)
    print("Random Forest Regression")
    print(r2_score(Y_test,Y_pred_rf))
    print(mean_absolute_error(Y_test,Y_pred_rf))
    print(np.sqrt(mean_squared_error(Y_test,Y_pred_rf)))
    return rf


# ## Model building using Gradient Boosting Regressor

# In[18]:


def gbr(X_train_std,X_test_std,Y_train,Y_test):
    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(X_train_std, Y_train)
    GradientBoostingRegressor(random_state=0)
    Y_pred_gbr=reg.predict(X_test_std[::])
    print("Gradient Boosting Regression")
    print(reg.score(X_test_std, Y_test))
    print(r2_score(Y_test,Y_pred_gbr))
    print(mean_absolute_error(Y_test,Y_pred_gbr))
    print(np.sqrt(mean_squared_error(Y_test,Y_pred_gbr)))
    return reg


# ## Model building using Lasso Regressor

# In[19]:


def lasso(X_train_std,X_test_std,Y_train,Y_test):
    lasso_reg = Lasso()
    lasso_reg.fit(X_train_std,Y_train)
    Y_pred_lasso=lasso_reg.predict(X_test_std[::])
    print("Lasso Regression")
    print(lasso_reg.score(X_test_std, Y_test))
    print(r2_score(Y_test,Y_pred_lasso))
    print(mean_absolute_error(Y_test,Y_pred_lasso))
    print(np.sqrt(mean_squared_error(Y_test,Y_pred_lasso)))
    return lasso_reg


# ## Hyper Parameter Tuning

# In[20]:


def hyperparam(X_train_std,X_test_std,Y_train,Y_test):

    # define models and parameters
    model = GradientBoostingRegressor()
    n_estimators = [10, 100, 1000]
    max_depth=range(1,31)
    min_samples_leaf=np.linspace(0.1, 1.0)
    max_features=["auto", "sqrt", "log2"]
    min_samples_split=np.linspace(0.1, 1.0, 10)

# define grid search
    grid = dict(n_estimators=n_estimators)

#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=101)

    grid_search_forest = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           scoring='r2',error_score=0,verbose=2,cv=2)

    grid_search_forest.fit(X_train_std, Y_train)

# summarize results
    print(f"Best: {grid_search_forest.best_score_:.3f} using {grid_search_forest.best_params_}")
    means = grid_search_forest.cv_results_['mean_test_score']
    stds = grid_search_forest.cv_results_['std_test_score']
    params = grid_search_forest.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean:.3f} ({stdev:.3f}) with: {param}")
    Y_pred_rf_grid=grid_search_forest.predict(X_test_std)
    r2_score(Y_test,Y_pred_rf_grid)
    return grid_search_forest


# In[ ]:




