#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier


# In[65]:


data = pd.read_csv("housing_data.csv")
X = data.drop('MEDV', axis=1)
y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[66]:


from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(max_depth = 1)
clf.fit(X_train, y_train)


# In[67]:


from sklearn.metrics import r2_score
train_p = clf.predict(X_train)
test_p = clf.predict(X_test)
print(r2_score(y_train, train_p))


# In[68]:


scorer = make_scorer(r2_score)
parameters = {'max_depth' : [1,2,3,4,5,6,7,8,9,10], 'max_leaf_nodes' : [None,2,3,4,5,6,7,8,9,10]}
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
grid_fit = grid_obj.fit(X_train, y_train)
best_clf = grid_fit.best_estimator_


# In[69]:


best_clf


# In[70]:


train_p = best_clf.predict(X_train)
test_p = best_clf.predict(X_test)
print("Train R2 Score", r2_score(y_train, train_p))
print("Test R2 Score", r2_score(y_test, test_p))


# In[71]:


best_clf.predict([[4,32,22]])


# In[ ]:




