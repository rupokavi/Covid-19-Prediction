#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, mean_squared_error,r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


# In[2]:


covid_data = pd.read_csv("E:\Covid19Prediction\Covid Dataset.csv")


# In[3]:


covid_data


# In[4]:


covid_data.columns


# In[5]:


# create a table with data missing 
missing_values=covid_data.isnull().sum() # missing values

percent_missing = covid_data.isnull().sum()/covid_data.shape[0]*100 # missing value %

value = {
    'missing_values ':missing_values,
    'percent_missing %':percent_missing  
}
frame=pd.DataFrame(value)
frame


# In[6]:


e=LabelEncoder()


# In[13]:


columns_to_encode = [
    'Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
       'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
       'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
       'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
       'Attended Large Gathering', 'Visited Public Exposed Places',
       'Family working in Public Exposed Places', 'Wearing Masks',
       'Sanitization from Market', 'COVID-19'
]

# Apply encoding to each column
for col in columns_to_encode:
    covid_data[col] = e.fit_transform(covid_data[col])


# In[14]:


covid_data.head()


# In[15]:


covid_data.hist(figsize=(20,15));


# In[17]:


print(covid_data['Wearing Masks'].value_counts())
sns.countplot(x='Wearing Masks',data=covid_data)


# In[18]:


print(covid_data['Sanitization from Market'].value_counts())
sns.countplot(x='Sanitization from Market',data=covid_data)


# In[19]:


covid_data=covid_data.drop('Wearing Masks',axis=1)
covid_data=covid_data.drop('Sanitization from Market',axis=1)


# In[20]:


covid_data.columns


# In[21]:


x=covid_data.drop('COVID-19',axis=1)
y=covid_data['COVID-19']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)


# # Logistic Regression

# In[30]:


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

y_pred1 = log_reg.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred1)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred1)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')


# # K-Nearest Neighbours
# 

# In[41]:


knn = KNeighborsClassifier(n_neighbors=5)  # You can choose the value of k (here it's 5)
knn.fit(x_train, y_train)

# Making predictions
y_pred2 = knn.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred2)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred2)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')


# # Random Forest

# In[43]:


rf = RandomForestClassifier(n_estimators=100, random_state=101)  # You can set n_estimators as desired
rf.fit(x_train, y_train)

# Making predictions
y_pred3 = rf.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred3)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred3)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')


# # Decision Tree

# In[44]:


dt = DecisionTreeClassifier(random_state=101)
dt.fit(x_train, y_train)

# Making predictions
y_pred4 = dt.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred4)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred4)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')


# # Support Vector Machine

# In[45]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)

# Making predictions
y_pred5= svm.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred5)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred5)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')


# # Naive bayes

# In[47]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

# Making predictions
y_pred6 = nb.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred6)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred6)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')


# # Gradient Boosting 

# In[48]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=101)
gb.fit(x_train, y_train)

# Making predictions
y_pred = gb.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')


# # Multi-Layer Perceptron Classifier

# In[49]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=101, max_iter=500)  # You can adjust the parameters as needed
mlp.fit(x_train, y_train)

# Making predictions
y_pred = mlp.predict(x_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')

