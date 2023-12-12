# credit-risk-classification
Challenge assignment for supervised learning

////////////////////////////////////////////
Credit Risk Analysis Report:
////////////////////////////////////////////

## Overview of the Analysis

  This purpose of this analysis was to see if we could create a model that we could hand information 
  about prospective or existing loans from clients and predict if they will end up being a good or bad loan 
  in the long-run. This would help the bank decide if they should purchase the loan from another lender or 
  offer the loan to a prospective client.
  
  The factors we looked at included 'loan_size', 'interest_rate', 'borrower_income', 'debt_to_income',
  'num_of_accounts', 'derogatory_marks', and 'total_debt'.

  The outcomes were listed as 'loan_status' with a '0' for healthy loans and a '1' for high-risk ones. 

  We ran a logarithmic analysis using the original data, but as there was not an even amount of healthy and high-risk 
  loans in the dataset, also decided to use a resampling method to duplicate a selection of the high-risk
  datapoints for the training of a second logarithmic model.

  This allowed for a higher overall accuracy and precision for the model and would be the one I reccomend 
  (Model 2).

## Results

* Machine Learning Model 1:
  * Overall Accuracy Score: 0.9520479254722232
  * Healthy Loan Precision Score: 1.00
  * High-risk Loan Precision Score: 0.85
  * Healthy Loan Recall Score: 0.99
  * High-risk Loan Precision Score: 0.91



* Machine Learning Model 2:
  * Overall Accuracy Score: 0.9936781215845847
  * Healthy Loan Precision Score: 1.00 
  * High-risk Loan Precision Score: 0.84
  * Healthy Loan Recall Score: 0.99
  * High-risk Loan Precision Score: 0.99

## Summary

When we added resampled data so we had more points for bad loans (Model 2), we were able to predict good and bad loans given the factors provided with an overall accuracy of 0.9936781215845847.

The model does much better at only marking helthy loans as healthy than it does at only marking bad loans as bad (based on percision scores). Although I would like the percison of bad loan labeling to be higher than 84%, it is better to mark a good loan as potentially worrying than to mark a bad loan as good.

Also, the second model accurately marks 99% of both the healthy and bad loans as what they are (given their recall scores). 

This model could be used by a company to do an inital screening of clinets for loans or for a pre-approval. If they pass with a "0", we can be fairly sure it will be a good loan.

However, if someone applies and gets a "1", we may want to have an underwriter manually review them due to regulations around offering loans and the possability of false positives being mixed in that could actually be good loans.

If the company has access to more acutal data points for bad loans, I would share that the model could be re-created to potentially be even better.


////////////////////////////////////////////
Sources for Code
////////////////////////////////////////////

Layout for assignment ipynb file came from starter file.

Specific sections directly using sources listed below:

--------------------------------------------------
Setup 
--------------------------------------------------

The following was provided in starter files:

    # Import the modules
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

--------------------------------------------------
Predict a Logistic Regression Model with Resampled Training Data
--------------------------------------------------

The following example:

    from collections import Counter
    from sklearn.datasets import make_classification
    from imblearn.over_sampling import RandomOverSampler
    X, y = make_classification(n_classes=2, class_sep=2,
    weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({1: 900, 0: 100})
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({0: 900, 1: 900})

From:
    
    https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html

Was used as a model for the following section:


Luna from AskBCS helped me with the following code:

    # Count the distinct values of the resampled labels data
    y_res["loan_status"].value_counts()
