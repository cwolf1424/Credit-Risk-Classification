# credit-risk-classification
Challenge assignment for supervised learning

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



