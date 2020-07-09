#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train=pd.read_csv("train_LZdllcl.csv")


# In[3]:


test=pd.read_csv("test_2umaH9m.csv")


# In[4]:


train.head(3)


# In[5]:


test.head(4)


# In[6]:


len(train)


# In[7]:


len(test)


# In[8]:


train.shape


# In[9]:


test.shape


# In[10]:


train.dtypes


# In[11]:


train.isnull().sum()/len(train)


# In[12]:


test.isnull().sum()/len(test)


# In[13]:


train.describe()


# In[14]:


test.describe()


# In[15]:


train.columns


# # EDA TO Understand the Data 

# In[16]:


sns.distplot(train['age'])  


# we see that the age is slightly postively skewed ,however it seems to be normally distributed

# In[17]:


sns.distplot(np.log(train['age']))


#  looks normall distributed after taking the mean

# In[18]:


sns.distplot(train['length_of_service'])


# The length of service skewed postively 

# In[19]:


train['gender'].value_counts()/len(train)*100


# The organisation is male dominated ,going forward we will see its effect

# In[20]:


train['KPIs_met >80%'].value_counts()/len(train)


# In[21]:


pd.crosstab(train['gender'],train['KPIs_met >80%'])/len(train)  


# ratio of kpi met or not for female is more than men .

# In[22]:


train['previous_year_rating'].value_counts().plot(kind='bar')


# The average rating of the organisation is 3.0

# In[23]:


sns.countplot(train['recruitment_channel'])


# Most of the employees have been recruited through other recuitment channels

# # combination of variables insights 

# In[24]:


pd.crosstab(train['previous_year_rating'],train['gender']
           ).plot(kind='bar',grid=False,stacked='True')


# Most of the female employees have been rated three & above

# In[25]:


pd.crosstab(train['is_promoted'],train['awards_won?']).plot(kind='bar',stacked=True)


# Employees who have been awarded before have more tendency to get promoted 

# In[26]:


#dependency of KPIs with Promotion

data = pd.crosstab(train['KPIs_met >80%'], train['is_promoted'])
data.div(data.sum(1).astype('float'), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 8), color = ['pink', 'darkred'])

plt.title('Dependency of KPIs in determining Promotion', fontsize = 30)
plt.xlabel('KPIs Met or Not', fontsize = 20)
plt.legend()
plt.show()


# Those who are completing the KPI's are mostly promoted 

# In[27]:


# checking dependency on previous years' ratings

data = pd.crosstab(train['previous_year_rating'], train['is_promoted'])
data.div(data.sum(1).astype('float'), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 8), color = ['violet', 'pink'])

plt.title('Dependency of Previous year Ratings in determining Promotion', fontsize = 30)
plt.xlabel('Different Ratings', fontsize = 20)
plt.legend()
plt.show()


# Here we can see that , that the propertion of  employees having more than 4 rating have been promoted  more ,so prevoius year rating is important for finding whether employee will be promoted or not 

# In[28]:


# checking which department got most number of promotions

data = pd.crosstab(train['department'], train['is_promoted'])
data.div(data.sum(1).astype('float'), axis = 0).plot(kind = 'bar', stacked = True, figsize = (20, 8), color = ['orange', 'lightgreen'])

plt.title('Dependency of Departments in determining Promotion of Employees', fontsize = 30)
plt.xlabel('Different Departments of the Company', fontsize = 20)
plt.legend()
plt.show()


# There is no biasness for promotion of employess based on department /

# In[29]:


# checking dependency of gender over promotion

data = pd.crosstab(train['gender'], train['is_promoted'])
data.div(data.sum(1).astype('float'), axis = 0).plot(kind = 'bar', stacked = True, figsize = (7, 5), color = ['green', 'red'])

plt.title('Dependency of Genders in determining Promotion of Employees', fontsize = 30)
plt.xlabel('Gender', fontsize = 20)
plt.legend()
plt.show()


# No baisness ,in promotion among gender ,however this thing was observed in employee rating

# 
# # DATA PREPROCESSING

# In[30]:


train.isnull().sum()


# In[31]:


test.isnull().sum()


# In[32]:


# filling missing values

train['education'].fillna(train['education'].mode()[0], inplace = True)
train['previous_year_rating'].fillna(1, inplace = True)


# In[33]:



train.isnull().sum()


# In[34]:


test['education'].fillna(test['education'].mode()[0],inplace=True)
test['previous_year_rating'].fillna(1,inplace=True)


# In[35]:


test.isnull().sum()


# In[36]:



train = train.drop(['employee_id'], axis = 1)

train.columns


# In[37]:


emp_idS = test['employee_id']
test = test.drop(['employee_id'], axis = 1)
test.columns


# In[38]:



x_test = test

x_test.columns


# In[39]:



# one hot encoding for the test set

x_test = pd.get_dummies(x_test)

x_test.columns


# In[40]:


# splitting the train set into dependent and independent sets

x = train.iloc[:, :-1]
y = train.iloc[:, -1]

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)


# In[41]:


y


# In[42]:


# one hot encoding for the train set

x = pd.get_dummies(x)

x.columns


# In[43]:


y.value_counts()


# In[44]:


#oversampling the dataset ,since it is imbalanced


# In[45]:


# !pip install imblearn


# In[46]:


# from imblearn.over_sampling import SMOTE

# x_sample, y_sample = SMOTE().fit_sample(x, y.values.ravel())

# x_sample = pd.DataFrame(x_sample)
# y_sample = pd.DataFrame(y_sample)

# # checking the sizes of the sample data
# print("Size of x-sample :", x_sample.shape)
# print("Size of y-sample :", y_sample.shape)


# In[47]:


# splitting x and y into train and validation sets

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("Shape of x_train: ", x_train.shape)
print("Shape of x_valid: ", x_valid.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_valid: ", y_valid.shape)


# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

rfc_pred = rfc.predict(x_test)

print("Training Accuracy :", rfc.score(x_train, y_train))

'''
print("Validation Accuracy :", rfc.score(x_valid, y_valid))

cm = confusion_matrix(y_valid, rfc_pred)
print(cm)

cr = classification_report(y_valid, rfc_pred)
print(cr)

apc = average_precision_score(y_valid, rfc_pred)
print("Average Precision Score :", apc)
'''


# In[49]:


from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etc.fit(x_train, y_train)

etc_pred = etc.predict(x_test)

print("Training Accuracy :", etc.score(x_train, y_train))
'''
print("Validation Accuracy :", etc.score(x_valid, y_valid))

cm = confusion_matrix(y_valid, etc_pred)
print(cm)

cr = classification_report(y_valid, etc_pred)
print(cr)

apc = average_precision_score(y_valid, etc_pred)
print("Average Precision Score :", apc)
'''

