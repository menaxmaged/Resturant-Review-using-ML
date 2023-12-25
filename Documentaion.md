# Restaurant Review Classification Program

This code is a simple restaurant review classification program using machine learning techniques.

## Libraries Used

The following libraries are imported for data processing and analysis:

- pandas: It is used for data manipulation and analysis.
- sklearn.metrics: It is used for measuring the performance of the trained model.

## Data Loading

The `Restaurant_Reviews.csv` dataset is loaded using pandas' `read_csv()` function. 

## Data Cleaning

The dataset has a column named '  Review', which is renamed to 'Review' for better access. Also, 'Liked' column is converted to a category data type.

## Preprocessing

The 'Review' column is preprocessed to get clean text for better model training. CountVectorizer from scikit-learn is used to create a bag of words.

### Machine Learning

The app uses the following machine learning models to predict the sentiment of a given review:

- Support Vector Machine (SVM): Assigned to Mena Maged.
- Naive Bayes: Assigned to Abdelrahman Ahmed Goda.
- Logistic Regression: Assigned to Marwa Salem.
- Decision tree Regression: Assigned to Sabry Salah.

Each model is trained on a the dataset and evaluated for accuracy using the remaining portion.
and each one of them in a sepacrated python file.

## User Input

The user is asked to input a review that is then processed and classified as positive or negative.

## Conclusion

This code is a simple example of a text classification model using Decision Tree Regressor for restaurant reviews. The model can be improved by using other machine learning models and better text preprocessing techniques.
