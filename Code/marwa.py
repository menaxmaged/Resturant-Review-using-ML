
import pandas as pd
from sklearn.metrics import accuracy_score

# Loading Data Set
df = pd.read_csv("Restaurant_Reviews.csv")


df["Liked"]=df["Liked"].astype("category")

"""**WORD** **Manpultation**"""

df[' Review']
df.rename(columns = {' Review':'Review'}, inplace = True) #small fix APPLY ONCE


X = df["Review"]
y = df["Liked"]

"""**Machine Learning**"""

from sklearn.feature_extraction.text import CountVectorizer

# Preprocess the data
#vectorizer = TfidfVectorizer(stop_words='english')
vectorizer = CountVectorizer()

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

def wordman(text):
 # Manpultation of the Entered words
 #lemword=lemmatize_sentence(text)
 
 vec_word = vectorizer.transform([text])
 return vec_word



"""MAIN APP """
print("Welcome to our Restaurant Review App")
print("Our App will predict if your review is possitive or negative")
print("Starting traing Using Logistic Regression ")

"""**Algorithm 1**"""
from sklearn.linear_model import LogisticRegression
# Train the logistic regression model
logistic = LogisticRegression()
logistic.fit(X_train, y_train)


# Predict on the test set and calculate accuracy
predicted = logistic.predict(X_test)
LR_score = accuracy_score(y_test, predicted)
print("Training Done with Accuracy: ",LR_score*100,"%")


####APP UI #####

text = input("Enter Your Review: ")
rate = logistic.predict(wordman(text))
print("Your Review is a Possitive Review") if rate == 1 else print("Your Review is a Negative Review")
