import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# importing news.csv
df=pd.read_csv("news.csv")
print(df.head())

# printing shape of news.csv
print(df.shape)

# printing the labels
labels=df.label
print(labels)

# spliting the data training and testing
X_train,X_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.25,random_state=7)

# Initialize the tfidfvectorize
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=7.0)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)

#DataFlair - Initialize a PassiveAggressiveClassifier
import xgboost as xg
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

print(confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))