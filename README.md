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

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores. 

Classification Report
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

Classification Report Resampled Data
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     75036
           1       0.99      0.99      0.99     75036

    accuracy                           0.99    150072
   macro avg       0.99      0.99      0.99    150072
weighted avg       0.99      0.99      0.99    150072


## Summary

Question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

Answer: The logistic regression model shows very good predictions for both the healthy loans classification (100%) and for the high-risk loans (88%).

Question: How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

Answer: Using oversampled data increased the f1-score of high-risk loan by 11% (88 to 99). The healthy loan percentage only decreased by 1% (now 99 instead of 100). Using the oversampled method is much more efficient in predicting loan repayment. Also note that the accuracy is the same for both methods (99%).

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
