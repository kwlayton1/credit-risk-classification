# Credit Risk Classification Report [Module 20]

## Overview of the Analysis (note that the next section will show the actual results):

#### The purpose of the analysis is to develop a logistic regression model (supervised machine learning) that predicts Credit Risk. The loan_status variable prediction is used to predict the Credit Risk. 

#### The analysis uses an input csv file with the following information: loan_size, interest_rate, borrower_income, debt_to_income, number_of_accounts, derogatory_marks and the total debt.  
#### There is also a loan_status column that reflects the classification established for the loan_status of each row. This contains a '0' (healthy loan) or a '1' (high-risk). 

#### Using the loan_status column for the 'labels' set (y) and the rest of the columns as the 'features' set (x). The balance of the labels variable y (target values) was checked using 'value_counts'  

#### The data was split using train_test_split (from sklearn).

### For Machine Learning Model 1 -> a Logistic Regression Model was built using the Original data (LogisticRegression, also from sklearn). Model was 'fit' using training data.

Predictions were created and saved on the testing data labels by using the testing feature data (X_test) and the fitted model.

The models's performance was evaluated using the accuracy score, a confusion matrix and classification report.

### For Machine Learning Model 2 -> predict a Logistic Regression Model using resampled training data.

#### As the target values were imbalanced on the original data - about 75% healthy loans (0) and 25% high-risk loans (1). The data was resampled using RandomOverSampler from the imbalanced-learn library. The resampled data now contains an equal number of samples from healthy and high risk loans (due to random samples of the high risk loans being resampled and added back into the high risk group). 

#### LogisticRegression classifier was used again with the resampled data to fit the model and make predictions. 

The models's performance was evaluated using the accuracy score, a confusion matrix and classification report.

## Results 

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

### Machine Learning Model 1: 
###   1-  99% Accuracy score. Measures the percentage of correctly classified instances out of the total instances in the dataset.
###   2- For 0:  100% Precision score. The ratio of true positives to the total number of predicted positives
###      For 1:   85% Precision score.
###   3- For 0:  99% Recall score.   The ratio of true positives to the total number of actual positives.
###      For 1:   85% Precision score.

####Classification Report for Model 1:
####              precision    recall  f1-score   support
####           0       1.00      0.99      1.00     18765
####           1       0.85      0.91      0.88       619
####    accuracy                           0.99     19384
####   macro avg       0.92      0.95      0.94     19384
####  weighted avg       0.99      0.99      0.99     19384

### Machine Learning Model 2:
###   1-  99% Accuracy score. Measures the percentage of correctly classified instances out of the total instances in the dataset.
###   2- For 0:  99% Precision score. The ratio of true positives to the total number of predicted positives
###      For 1:  99% Precision score.
###   3- For 0:  99% Recall score.   The ratio of true positives to the total number of actual positives.
###      For 1:   99% Precision score.
###  
####Classification Report Resampled Data
####              precision    recall  f1-score   support
####           0       0.99      0.99      0.99     75036
####           1       0.99      0.99      0.99     75036
####
####v   accuracy                           0.99    150072
####   macro avg       0.99      0.99      0.99    150072
####weighted avg       0.99      0.99      0.99    150072


## Summary

### Model 1, the logistic regression model using original data shows good predictions for both the healthy loans classification (100%) and for the high-risk loans (88%).

### Model 2, I abxolutely recommend the logistic regression model, fit with oversampled data. The macro-averaged precision, recall, and F1-score are all 0.99, indicating that the model's performance is excellent, and it has achieved high precision, recall, and F1-score for both classes. This means that the model is making accurate predictions for both positive and negative instances.

