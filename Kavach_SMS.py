

import pandas as pd
import numpy as np
df = pd.read_csv('spam.csv', on_bad_lines='skip', encoding='latin1')
df.head(5)
df.shape
df.info()
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)




df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)



from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()



df['target'] = encoder.fit_transform(df['target'])
df.head()
df.isnull().sum()
#check for the duplicate values
df = df.drop_duplicates(keep='first')



df.duplicated().sum()




df['target'].value_counts()


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


df['target'].value_counts()
import nltk
get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')
#lets find the number of character in each term
df['num_characters'] = df['text'].apply(len)
df.head()
#to find the number of words we will use tokenization 
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

df.head()




#lets find the number of sentences
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df.head()
df[['num_characters','num_words','num_sentences']].describe()

#let us start with the data visualization 
import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')

plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')

#begin with the data preprocessing
from nltk.corpus import stopwords
import string
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

#lower case problem is solved
#special character is solved
#removing stop words

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('dancing')
transform_text('I loved the youtube lectures on Machine Learning? How about you?')
df['text'].apply(transform_text) 
df['transformed_text'] = df['text'].apply(transform_text)


#create the word cloud
from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size =5, background_color = 'white')

spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


plt.imshow(spam_wc)




ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))



plt.imshow(ham_wc)



spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


len(spam_corpus)




from collections import Counter
Counter(spam_corpus).most_common(30)



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['transformed_text']).toarray()
#x = cv.fit_transform(df['transformed_text']).toarray()
x.shape

y = df['target'].values
y

from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)



from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()



gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))



mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))



bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))



