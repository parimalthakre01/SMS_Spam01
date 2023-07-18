#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[7]:


df = pd.read_csv('spam.csv', on_bad_lines='skip', encoding='latin1')


# In[8]:


df.head(5)


# In[9]:


df.shape


# In[10]:


#1. data cleaning
#2. EDA
#3. Text preprocessing
#4. Model Building
#5. Evaluation 
#6. Improvement
#7. Website
#8. Deploy


# In[11]:


df.info()


# In[12]:


df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)


# In[13]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[14]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[15]:


df['target'] = encoder.fit_transform(df['target'])


# In[16]:


df.head()


# In[17]:


df.isnull().sum()


# In[18]:


#check for the duplicate values
df = df.drop_duplicates(keep='first')


# In[19]:


df.duplicated().sum()


# In[20]:


df['target'].value_counts()


# In[21]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[22]:


df['target'].value_counts()


# In[23]:


import nltk


# In[24]:


get_ipython().system('pip install nltk')


# In[25]:


import nltk


# In[26]:


nltk.download('punkt')


# In[27]:


#lets find the number of character in each term
df['num_characters'] = df['text'].apply(len)


# In[28]:


df.head()


# In[29]:


#to find the number of words we will use tokenization 
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[30]:


df.head()


# In[31]:


#lets find the number of sentences
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[32]:


df.head()


# In[33]:


df[['num_characters','num_words','num_sentences']].describe()


# In[34]:


#let us start with the data visualization 
import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[35]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[36]:


#begin with the data preprocessing
from nltk.corpus import stopwords
import string


# In[37]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[38]:


transform_text('Hi how are you Nitish?')


# In[39]:


#lower case problem is solved
#special character is solved
#removing stop words


# In[40]:


transform_text('Hi  he him how are you?')


# In[41]:


stopwords.words('english')


# In[42]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('dancing')


# In[43]:


transform_text('I loved the youtube lectures on Machine Learning? How about you?')


# In[57]:


df['text'].apply(transform_text) 


# In[45]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[46]:


df.head()


# In[47]:


#create the word cloud
from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size =5, background_color = 'white')


# In[48]:


spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[49]:


plt.imshow(spam_wc)


# In[50]:


ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[51]:


plt.imshow(ham_wc)


# In[52]:


spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[53]:


len(spam_corpus)


# In[54]:


from collections import Counter
Counter(spam_corpus).most_common(30)


# In[92]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['transformed_text']).toarray()
#x = cv.fit_transform(df['transformed_text']).toarray()
x.shape


# In[93]:


y = df['target'].values
y


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)


# In[96]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[97]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[98]:


gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[99]:


mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[100]:


bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[101]:


#tfidf-->mnb


# In[103]:


#import pickle
#pickle.dump(tfidf,open('vectorizer.pkl','wb'))
#pickle.dump(mnb,open('model.pkl','wb'))


# In[104]:





# In[ ]:




