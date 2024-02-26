from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn import model_selection, preprocessing, svm, metrics
import seaborn as sn

# Specify file location and encoding parameters
file_path = r'C:/Users/parimal/OneDrive/Desktop/Spam_dataset/spam.csv'
csv_encoding = 'latin-1'  # Try using 'latin-1' encoding

# Read the CSV file into a DataFrame
raw_data = pd.read_csv(file_path, encoding=csv_encoding, engine='python')

# Create a new DataFrame to store the relevant columns
data = pd.DataFrame()
data['target'] = raw_data['v1']
data['text'] = raw_data['v2']

ham = [i for i in data['target'] if i == 'ham']
spam = [i for i in data['target'] if i == 'spam']
len(ham), len(spam)

data['target'] = data['target'].map({'ham': 0, 'spam': 1})

data.replace(r'\b\w{1,4}\b', '', regex=True, inplace=True)

# Initialize CountVectorizer and transform the text data
vectorizer = CountVectorizer()
vec = vectorizer.fit_transform(data['text'])  # Transform text data

# Convert the sparse matrix to a dense array
dense_array = vec.toarray()

# Create a new DataFrame for the encoded text
encoded_text_df = pd.DataFrame(dense_array, columns=vectorizer.get_feature_names_out())

# Concatenate the encoded text DataFrame with the original data
data = pd.concat([data, encoded_text_df], axis=1)

# Train the SVM model
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(encoded_text_df, data['target'])

# Function to preprocess text
def preprocess_text(text):
    # Process the input text
    text = re.sub(r'\b\w{1,4}\b', '', text)  # Remove short words
    return text

# Function to predict whether a message is spam or not spam
# Function to predict whether a message is spam or not spam
def predict_spam(input_text):
    # Preprocess the input text
    processed_text = preprocess_text(input_text)
    # Transform the input text
    input_vec = vectorizer.transform([processed_text])
    # Convert the sparse input data to dense format
    input_vec_dense = input_vec.toarray()
    # Predict whether the message is spam or not spam
    prediction = SVM.predict(input_vec_dense)
    if prediction[0] == 0:
        return "Not Spam"
    else:
        return "Spam"

# Take input text from the user
user_input = input("Enter the text message: ")

# Predict whether the message is spam or not spam
prediction_result = predict_spam(user_input)
print("Prediction:", prediction_result)
