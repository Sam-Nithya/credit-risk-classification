# credit-risk-classification
Overview

This project aims to develop a machine learning model that can predict the credit risk of loans. The goal is to use historical lending data to classify loans as either healthy (low risk) or high risk of defaulting. We used a logistic regression model to train the data and evaluate its performance in predicting both labels.





Files

credit_risk_classification.ipynb: Jupyter notebook containing the code to load, preprocess, and model the data.
lending_data.csv: The dataset used to train the model. This file contains information on various loans, including the loan status (healthy or high-risk).
Data Preprocessing

The dataset was loaded from the Resources/lending_data.csv file. The following steps were performed to prepare the data for modeling:




Data Splitting:
The data was split into features (X) and labels (y), where y represents the loan status (0 for healthy and 1 for high-risk).
The data was then split into training and testing datasets using train_test_split from sklearn.
Model Training:
A Logistic Regression model was instantiated and trained using the training dataset (X_train and y_train).
The model was then used to predict loan statuses on the testing dataset (X_test), and the predictions were compared to actual labels.
Model Evaluation

Confusion Matrix:
The confusion matrix for the model's predictions is as follows:

[[14924    77]  # Healthy loans (label 0) predicted as healthy, predicted as high-risk
 [   31   476]]  # High-risk loans (label 1) predicted as healthy, predicted as high-risk





 
Classification Report:
The classification report for the model is as follows:

              precision    recall  f1-score   support
           0       1.00      0.99      1.00     15001
           1       0.86      0.94      0.90       507
    accuracy                           0.99     15508
   macro avg       0.93      0.97      0.95     15508
weighted avg       0.99      0.99      0.99     15508


Key Metrics:
Healthy loans (label 0):
Precision: 1.00
Recall: 0.99
F1-Score: 1.00
High-risk loans (label 1):
Precision: 0.86
Recall: 0.94
F1-Score: 0.90
Overall Performance:
Accuracy: 0.99
Macro Average: Precision: 0.93, Recall: 0.97, F1-Score: 0.95
Weighted Average: Precision: 0.99, Recall: 0.99, F1-Score: 0.99
Analysis:
The logistic regression model performs very well in predicting healthy loans (label 0) with near-perfect precision (1.00) and recall (0.99).
The model performs reasonably well on high-risk loans (label 1) with a recall of 0.94, meaning it identifies 94% of high-risk loans. However, there is some room for improvement in precision (0.86), as 14% of healthy loans are misclassified as high-risk.
The overall accuracy of 99% indicates that the model is robust, with a small number of misclassifications. The weighted averages show that the model excels in both classes, particularly on the healthy loan class.






Conclusion

The logistic regression model is highly effective at predicting both healthy and high-risk loans. While it performs excellently in predicting healthy loans, there is a minor trade-off in the prediction of high-risk loans due to slightly lower precision. However, the model still correctly identifies 94% of high-risk loans, making it a strong tool for credit risk classification.

