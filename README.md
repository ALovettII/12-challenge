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
![Evaluating the Original Data](https://github.com/ALovettII/12-challenge/blob/main/Resources/)
![Evaluating the Oversampled Data](https://github.com/ALovettII/12-challenge/blob/main/Resources/)
