#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Csv file of titanic dataset, In this file using following columns build a model to predict if person would survive or not.
#1. Pclass, 2. Sex, 3. Age, 4. Fare
#Calculate score of your model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
dataset=pd.read_csv(r'C:\Users\HP\Downloads\Python materials\titanic-training-data.csv')
dataset


# In[2]:


datanew=dataset.drop(['PassengerId','Name','SibSp','Ticket','Cabin','Embarked'],axis=1) #Axis=0 for detecting rows and 1 for cols


# In[3]:


datanew


# In[4]:


inputs=datanew.drop('Survived',axis='columns')


# In[5]:


target=datanew['Survived']


# In[6]:


inputs.isnull().sum()


# In[7]:


#We want to treat the 177 missing values of Age and hence we see if age is related to Parch or not. so boxplot.
sns.boxplot(x='Parch',y='Age',data=inputs,palette='rainbow') # 0 Parch - 30 mean age which means 0 Parch can be replaced by
#30 (mean age) if they have missing value.


# In[8]:


parch=inputs.groupby(inputs['Parch'])
parch.mean()


# In[9]:


#so where Parch = 0 and age is missing, replace that value with mean age corrosponding to Parch 0 i/e 32
def age1(col):
    Age=col[0]
    Parch=col[1]
    if pd.isnull(Age):
        if parch==0:
            return 32
        elif parch==1:
            return 24
        elif parch==2:
            return 17
        elif parch ==3:
            return 33
        elif parch==4:
            return 44
        elif parch ==5:
            return 39
        else:
            return 43
    else:
            return Age


# In[10]:


inputs['Age']=inputs[['Age','Parch']].apply(age1,axis=1) #applying ifelse defined above


# In[11]:


from sklearn.preprocessing import LabelEncoder
le_Sex=LabelEncoder()


# In[12]:


inputs['Sex_n']=le_Sex.fit_transform(inputs['Sex'])


# In[13]:


inputs


# In[14]:


inputs_n=inputs.drop(['Sex'],axis='columns')
inputs_n


# In[15]:


target


# In[16]:


inputs_n.isnull().sum()


# In[17]:


from sklearn import tree
model=tree.DecisionTreeClassifier()


# In[18]:


model.fit(inputs_n,target)


# In[19]:


model.score(inputs_n,target)


# In[20]:


model.predict([[1,27,0,72.3,1]])


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs_n,target,test_size=.2,random_state=91)


# In[23]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)


# In[24]:


classifier.score(inputs_n,target)


# In[25]:


model.predict([[1,27,0,72.3,1]])


# In[ ]:




