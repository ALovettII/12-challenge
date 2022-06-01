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
Open a terminal window, activate the `dev` virutal enviroment, and run the following commands:

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

When a financial institution provides a loan to an individual there is an inherent risk of default (i.e. loanee failing to pay the loan back). The purpose of the credit-risk analysis in this project is to predict whether the requested loan is a healthy (low-risk) loan or a high-risk loan. Using various machine learning techniques and information (features) about the loanee we can try to predict the risk associated with each loan.

However, credit risk poses a classification problem that’s often imbalanced. This is because healthy loans easily outnumber risky loans; thus leading to biased algorithms that are more adept at predicting the larger class than the minority. In order to compensate for this bias, the minority class is oversampled, inflating the data until the two are balanced. An example of this bias is displayed in the project.

### Basic information about the data in this project
* Financial dataset used for analysis: 
    * Historical lending activity from a peer-to-peer lending services company
* Predicted Variable:
    * Both the `0` (healthy loan) and `1` (high-risk loan)
* Value Counts for each variable:
    * `0`: 75,036
    * `1`: 2,500
* Value Counts for each class after resampling:
    * `0`: 56,271
    * `1`: 56,271
* Loanee Features:
    * loan_size
    * interest_rate
    * borrower_income
    * debt_to_income
    * num_of_accounts
    * derogatory_marks
    * total_debt
    
### Stages of the Machine Learning Process
1. Split the Data into Training and Testing Sets:
2. Create a Logistic Regression Model with the Original Data
3. Evaluate the model’s performance
4. Predict a Logistic Regression Model with Resampled Training Data:
5. Evaluate the model’s performance

### Methods
This project contains two methods in the analysis:
    1. `LogisticRegression`
    2. `RandomOverSampler` 
 
The `LogisticRegression` algorithm is considered one of the most universal and capable classification algorithms. The model is designed to predict discrete outcomes (i.e. a range of outcomes). We use this model to assess multiple variables in order to make a predictions about the loan status. If the sample of data is determined by the model to have a high degree of probability to belong to a class, it assigns the sample to that class.

The `RandomOverSampler` randomly selects instances of the minority class and add them to the training set until both the majority and minority classes are equal. This technique compensates for the problems arising from imbalanced classes. With this method, the model is able to train on a balanced dataset and thus able to predict new data with a relatively balanced accuracy.


## Results

* `LogisticRegression` Model 1 Scores:
    * Accuracy: (The differences between its predicted and its actual values)
        * 95.20%
    * Precision: (How confident we are that the model correctly made the positive predictions)
        * `0`: 1.00
        * `1`: .85
        
    * Recall: (Number of actually fraudulent transactions that the model correctly classified as fraudulent)
        * `0`: .99
        * `1`: .91
        
* Resampled `LogisticRegression` Model 2 Scores:
    * Accuracy: 99.37%
    * Precision:
        * `0`: 1.00
        * `1`: .84
    * Recall: 
        * `0`: .99
        * `1`: .99


## Summary

The best performing machine learning was Model 2, the Logoristuc Regression model trained on the randomly oversampled dataset. The balanced accuracy score has rose from 95.20% from Model 1 to 99.37% in Model 2. The harmonic mean for the `0` or 'healthy' class rose from 88% to 91%, and the `1` or 'high-risk' class remained at 1.00 for both models. This inidicates a higher percentage of true positives over total positive as well as over true positives and false negatives. 

The accuracy of our model bears great consequence on the well-being of the company. If 'high-risk' loans get mislabeled as 'healthy' loans, the company bears the default risk from loans that should've been denied. In order to protect the company from this risk, it is important that 'high-risk' loans are correctly assessed. Model 1 falsely predicted 56 instances of 'high-risk' loans as 'healthy', creating a large amount of risk for the company; however, the second model saw this happen in only 4 instances. The drastic reduction in false positives porportionally decreases default risk in the loans it distributes. The accuracy of this model can be directly attributed to the random oversampling method applied to the dataset. The result of this oversampling method was a balanced prediction accuracy for both classes. 

After asessing the performance of each machine learning model on the prediction of the degree of loan risk, the reccomended machine-learning model is Model 2, the model trained on randomly oversampled data. This model poses the lowest degree of risk to the company in its predictions. However, I would also suggest more training performed on the model before put into use to ensure the lowest number of false positives predictions possible.
 
 
## Contributors
Created by Arthur Lovett