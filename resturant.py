# -*- coding: utf-8 -*-
"""Resturant.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DZVvgwW20A3XfWIJyiBwPXzcvGuV2dAw

Importing Section
"""

import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

stop_words=stopwords.words("english")
lemmatizer = WordNetLemmatizer()

"""

 ```
# Loading Data Set
```

"""

df = pd.read_csv("Restaurant_Reviews.csv")

df.head()

df["Liked"]=df["Liked"].astype("category")

"""**WORD** **Manpultation**"""

df[' Review']
df.rename(columns = {' Review':'Review'}, inplace = True) #small fix APPLY ONCE

df['lemmatized_reviews']=df['Review'] # applying the function to the dataset to get clean text
df.head()

X = df["lemmatized_reviews"]
y = df["Liked"]


"""**Machine Learning**"""

from sklearn.feature_extraction.text import CountVectorizer

# Preprocess the data
#vectorizer = TfidfVectorizer(stop_words='english')
vectorizer = CountVectorizer()

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=45)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

def wordman(text):
 #lemword=lemmatize_sentence(text)
 
 tfidfword = vectorizer.transform([text])
 print("You Entered:"+ text)
 return tfidfword

"""**Algorithm 1**"""

from sklearn.linear_model import LogisticRegression

# Train the logistic regression model
logistic = LogisticRegression()
model = logistic.fit(X_train, y_train)

#Testing The Work 
from sklearn.metrics import accuracy_score

# Predict on the test set and calculate accuracy
predicted = model.predict(X_test)
LR_score = accuracy_score(y_test, predicted)

# Print the accuracy
print("Accuracy: ", score)

"""**Algorithm2**"""

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
ypredict = nb.predict(X_test)
NB_Score = accuracy_score(y_test, ypredict)
print(NB_Score)

"""MAIN APP """

text = "I do not love the place"
rate = nb.predict(wordman(text))
print("Possitive Review") if rate == 1 else print("Negative Review")