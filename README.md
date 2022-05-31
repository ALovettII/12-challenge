# 12-challenge
 Using various techniques to train/evaluate models with imbalanced classes -  identifying the creditworthiness of borrowers


## Technologies

* import numpy as np
* import pandas as pd
* from pathlib import Path
* from sklearn.metrics import balanced_accuracy_score
* from sklearn.metrics import confusion_matrix
* from imblearn.metrics import classification_report_imbalanced


## Installation Guide

 Using the Conda package manager: [My GitHub Project](https://github.com/ALovettII/12-challenge.git)


### Install the required libraries
 Open a terminal window, activate your `dev` virutal enviroment, and run the following commands:

```Python
# Install imbalance-learn
conda install -c conda-forge imbalanced-learn

# Install PyDotPlus
conda install -c conda-forge pydotplus
```

## Usage

* Running this program will allow the following:
* Split the Data into Training and Testing Sets
* Create a Logistic Regression Model with the Original Data
* Predict a Logistic Regression Model with Resampled Training Data

The program will yield the following results:
![Evaluating the Original Data](https://github.com/ALovettII/12-challenge/blob/main/Resources/og-data_eval.png)
![Evaluating the Oversampled Data](https://github.com/ALovettII/12-challenge/blob/main/Resources/os-data_eval.png)

The data used for analysis may be modified by changing the `df_lending_data` variable 
*(the written interpretations of the model evaluations will be rendered obsolete)*


## Overview of the Analysis

 When a financial institution provides a loan to an individual there is an associated risk of default (i.e. the loanee failing to pay the loan back). The purpose of the credit-risk analysis in this project is to predict whether the requested loan is a healthy/low-risk loan or a high-risk loan. Using machine learning techniques and information (features) about the loanee we can try to predict the risk involved with each loan.

However, credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans; thus leading to biased algorithms that are more adept at predicting the larger class than the minority. In order to compensate for this bias, the minority class is oversampled, inflating the data until the two are balanced.

### Basic information about the data in this project:
* Financial dataset used for analysis: 
    * historical lending activity from a peer-to-peer lending services company
* Predicted Variable:
    * Both the `0` (healthy loan) and `1` (high-risk loan)
    
* Value Counts for each variable:
    * `0`: 75036
    * `1`: 2500
* Loanee Features:
    * loan_size
    * interest_rate
    * borrower_income
    * debt_to_income
    * num_of_accounts
    * derogatory_marks
    * total_debt
    
### Credit Risk Analysis: Stages of the Machine Learning Process
1. Split the Data into Training and Testing Sets:
    * Reading in the lending data from a CSV file
    * Creating the labels set (y) from the `loan_status` column
    * Creating the features (X) DataFrame from the remaining columns
    * Check the balance of the labels variable (y) by using the value_counts function
    
2. Split the data into training and testing datasets by using train_test_split:
    * Create a Logistic Regression Model with the Original Data
    * Employ your knowledge of logistic regression to complete the following steps:
        * Fit a logistic regression model by using the training data (X_train and y_train).
        * Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.
    * Evaluate the model’s performance by doing the following:
        * Calculate the accuracy score of the model.
        * Generate a confusion matrix.
        * Print the classification report.
    * Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

3. Predict a Logistic Regression Model with Resampled Training Data:
    * Use the RandomOverSampler module from the imbalanced-learn library to resample the data
    * Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.
    * Evaluate the model’s performance by doing the following:
        * Calculate the accuracy score of the model.
        * Generate a confusion matrix.
        * Print the classification report.
    * Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

This project contains two methods in the analysis:
    1. `LogisticRegression`
    2. `RandomOverSampler` 
 
The `LogisticRegression` algorithm is considered one of the most universal and capable classification algorithms. The model is designed to predict discrete outcomes (i.e. a range of outcomes). We use this model to assess multiple variables in order to make a predictions about the loan status. If the sample of data is determined by the model to have a high degree of probability to belong to a class, it assigns the sample to that class.

The `RandomOverSampler` randomly selects instances of the minority class and add them to the training set until we balance. the majority and minority classes. This technique compensates for the problems arising from imbalanced classes. With this method, the model is able to train on a balanced dataset and thus able to predict new data with a relatively balanced accuracy.

## Results

* `LogisticRegression` Model 1 Scores:
    * Accuracy: 95.20%
    * Precision: 
        * `0`: 1.00
        * `1`: .85
    * Recall
        * `0`: .99
        * `1`: .91
        
* Resampled `LogisticRegression` Model 2 Scores:
    * Accuracy: 99.37%
    * Precision:
        * `0`: 1.00
        * `1`: .84
    * Recall
        * `0`: .99
        * `1`: .99


## Contributors
 Created by Arthur Lovett