#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("diabetes.csv")


# In[2]:


df.head()


# In[3]:


df.shape


# In[4]:


# find the count of missing values in each column
df.isnull().sum()


# In[5]:


# Selecting all rows and all columns except the last one for features and  convert into NumPy arrays
X=df.iloc[:,:-1].to_numpy()
# Selecting all rows and the last column for the target and convert  into NumPy arrays
y=df.iloc[:,-1].to_numpy()


# In[6]:


from sklearn.model_selection import train_test_split
'''
y-target variable
X-input columns
random_state=0 ensures that the data split will be the same every time you run the code with the same input data.
test size=20%
train size=80%
'''
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[8]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy",random_state=0)
clf.fit(X_train,y_train)


# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[10]:


clf.set_params(max_depth=3)


# In[11]:


clf.fit(X_train,y_train)
plt.figure(figsize=(20,10))
plot_tree(clf,feature_names=['Glucose','BMI'],class_names=['No','Yes'])
plt.show()


# In[12]:


predictions=clf.predict(X_test)


# In[13]:


clf.predict([[90,20],[200,30]])


# In[14]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X_train,y_train,cv=5,scoring='accuracy')
accuracy=scores.mean()#calculates the mean accuracy across the 5 folds of cross-validation.
accuracy


# In[15]:


from sklearn import metrics
cf=metrics.confusion_matrix(y_test,predictions)
cf


# In[16]:


tp=cf[1][1]
tn=cf[0][0]
fp=cf[0][1]
fn=cf[1][0]
print(f"tp:{tp}, tn:{tn},fp:{fp},fn:{fn}")


# In[17]:


print("accuracy",metrics.accuracy_score(y_test,predictions))


# In[18]:


print("Precision",metrics.precision_score(y_test,predictions))


# In[19]:


print("Recall",metrics.recall_score(y_test,predictions))


# In[20]:


feature_importances = clf.feature_importances_
print("Feature importances:",feature_importances)


# In[ ]:




