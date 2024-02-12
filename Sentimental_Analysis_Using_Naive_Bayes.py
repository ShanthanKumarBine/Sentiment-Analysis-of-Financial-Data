#!/usr/bin/env python
# coding: utf-8

# # Sentiment analysis using Naive Bayes

# ## - Shanthan Kumar Bine

# In[75]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### These are standard import statements in Python for working with data analysis and visualization libraries.

# In[76]:


df = pd.read_csv(r'C:\Users\Shant\Downloads\data.csv')


# In[77]:


df.head()


# ### This code reads in a dataset from a CSV file 

# In[78]:


df.isnull().sum()


# ### This code uses the isnull() method to check for missing values in each column of the df DataFrame. 

# In[79]:


df.info()


# ### This code displays a summary of the df DataFrame using the .info() method from the pandas library.

# In[80]:


df.duplicated().sum()


# ### This code uses the duplicated() method to check for duplicate rows in the df DataFrame.

# In[81]:


df.drop_duplicates(inplace=True)


# In[82]:


df.shape


# In[83]:


df['Sentiment'].value_counts()


# In[84]:


neutral_count = 0
positive_count = 0
negative_count = 0

for i in df['Sentiment']:
    if i == 'neutral':
        neutral_count += 1
    elif i == 'positive':
        positive_count += 1
    elif i == 'negative':
        negative_count += 1

total_count = len(df)
neutral_percent = format((neutral_count / total_count) * 100, '.2f')
positive_percent = format((positive_count / total_count) * 100, '.2f')
negative_percent = format((negative_count / total_count) * 100, '.2f')

print(f'{neutral_percent}% people have neutral sentiment')
print(f'{positive_percent}% people have positive sentiment')
print(f'{negative_percent}% people have negative sentiment')


# ## This code calculates the number and percentage of sentences in the DataFrame that have each of the three sentiments: neutral, positive, and negative.

# In[85]:


sns.countplot(x='Sentiment',data=df)


# ### This code creates a countplot using the countplot() function from the seaborn library. 

# In[86]:


X = df['Sentence']
Y = df['Sentiment']


# In[87]:


from sklearn.model_selection import train_test_split


# ## This code imports the train_test_split() function from the sklearn.model_selection module. 

# In[88]:


X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# In[89]:


from sklearn.feature_extraction.text import CountVectorizer


# ### Conversion of text to vector

# In[90]:


v = CountVectorizer(stop_words='english')
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)


# In[91]:


from sklearn.naive_bayes import MultinomialNB


# In[92]:


multNB = MultinomialNB()
multNB.fit(X_train,Y_train)


# In[93]:


Y_pred = multNB.predict(X_test)


# ### Model performance

# In[94]:


from sklearn.metrics import accuracy_score


# In[95]:


accuracy_score(Y_test,Y_pred)


# In[ ]:




