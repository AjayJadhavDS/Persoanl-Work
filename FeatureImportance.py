#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[2]:


import pandas as pd
df = pd.read_csv('C:/Users/Downloads/DatasetsGear7.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# # ReliefF

# In[5]:


features, labels = df.drop('Class', axis=1).values, df['Class'].values

clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
                    RandomForestClassifier(n_estimators=100))


# In[6]:


#print(np.mean(cross_val_score(clf, features, labels)))


# In[7]:


from sklearn.feature_selection import mutual_info_classif


# # mutual info

# In[8]:


newDf = pd.DataFrame({'features': df.drop('Class', axis=1).columns, 'importance':mutual_info_classif(features, labels)})
newDf.head()


# In[9]:


newDf.plot.bar(x='features', y='importance', rot=0)


# # NB SVM DT RNN

# In[37]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df['Class'])


# In[38]:


X_test, X_val, y_test, y_val = train_test_split(X_test, y_test)


# In[42]:


print('training set ',X_train.shape)


# In[43]:


print('testing set ',X_test.shape)


# In[40]:


print('validattion set ',X_val.shape)


# In[ ]:


X_train


# In[16]:


y_train.shape


# In[45]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
dtree_model = DecisionTreeClassifier().fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
  
# creating a confusion matrix 
print('Accuracy of Decision tree model test set is  ',accuracy_score(y_test, dtree_predictions)*100)

print('Accuracy of Decision tree model on validation set is  ',accuracy_score(y_val, dtree_model.predict(X_val))*100)


# In[46]:


from sklearn.svm import SVC 
svm_model_linear = SVC().fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 

print('Accuracy of SVM model is on training set ',accuracy_score(y_test, svm_predictions)*100)

print('Accuracy of SVM model on validation set is  ',accuracy_score(y_val, svm_model_linear.predict(X_val))*100)


# In[47]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 


print('Accuracy of Naive bayes model on test set is  ',accuracy*100)

print('Accuracy of Decision tree model on validation set is  ',accuracy_score(y_val, gnb.predict(X_val))*100)


# In[27]:


from keras.utils import np_utils

dummy_y = np_utils.to_categorical(y_train)


# In[36]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(dummy_y.shape[1], activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,dummy_y ,epochs=30, batch_size=64)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




