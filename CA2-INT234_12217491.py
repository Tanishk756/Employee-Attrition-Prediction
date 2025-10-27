#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


#Load Dataset

df = pd.read_csv("C:/Users/Dell/Desktop/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()


# In[3]:


#Check for Missing Values

df.isnull().sum()


# In[4]:


#Basic Info

df.info()
df.describe()


# In[5]:


#Target Variable Distribution

sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition Distribution")


# In[6]:


#Attrition vs Department

sns.countplot(x='Department', hue='Attrition', data=df)


# In[7]:


#Attrition vs Overtime

sns.countplot(x='OverTime', hue='Attrition', data=df)


# In[8]:


#Encoding Categorical Variables

le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])


# In[9]:


#Feature Scaling

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Attrition', axis=1))
X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
y = df['Attrition']


# In[10]:


#Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


#Model 1: Decision Tree Classifier

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)


# In[17]:


#Model Evaluation

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))


# In[18]:


#Model 2: Support Vector Machine (SVM)

svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)


# In[19]:


#Model Evaluation

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))


# In[ ]:




