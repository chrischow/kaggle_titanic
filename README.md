# Kaggle Competition - Titanic Dataset
This Kaggle competition requires participants to predict whether passengers will survive.

## Introduction
This readme describes the entire process I used for the Kaggle Titanic competition. The notebooks are named in pseudo-chronological order. I say "pseudo" because whenever I revisited a process, I edited the relevant existing notebook without creating a new one. Hence, some code and results in earlier notebooks may not have been used in the next few notebooks, but were used at some point in the entire project.  
  
## Exploratory Data Analysis
First, I performed exploratory data analysis (EDA) on all features in the dataset ([1. EDA.ipynb](https://github.com/chrischow/kaggle_titanic/blob/master/Notebooks/1.%20EDA.ipynb)). Concurrently, I cleaned the data and generated new features.
  
## Modelling
  
### Random Forest (`sklearn`)
Next, I experimented with the `sklearn` implementation of Random Forest ([2. Random Forest - Low Depth.ipynb](https://github.com/chrischow/kaggle_titanic/blob/master/Notebooks/2.%20Random%20Forest%20-%20Low%20Depth.ipynb)). By the final iteration in the optimisation process, the accuracy was **82.47%**. However, the `sklearn` implementation (`RandomForestClassifier`) was extremely slow. Hence, I switched over to the `lightgbm` implementation of the Random Forest algorithm. 
  
### Random Forest (`lightgbm`)
The `lightgbm` implementation of the Random Forest algorithm was **much** faster. The subsequent three notebooks captured my attempts at figuring out the best sequence for the tuning of parameters. The best approach was to:  
  
1. Tune the number of trees/estimators
2. Perform Recursive Feature Elimination (RFE) with cross-validation (CV)
3. Tune the proportion of observations sampled
4. Tune the maximum features used per split
5. Tune the minimum samples in each terminal node
6. Tune alpha, the L1 regularisation parameter
7. Tune lambda, the L1 regularisation parameter
  
This tuning sequence resulted in a 5-fold CV test score of **83.46%**.  
  
### Regenerating Features
Then, I realised that I made a mistake in the feature generation process. I had generated encoded features using the entire training set. This resulted in contamination, which meant that the pseudo-test sets in CV contained information about the target, and this could have inflated the test scores. Instead, I should have (1) performed data cleaning on the competition training and test sets together, (2) encoded the training set while saving the mappings, and (3) performed mappings on the test set.  Hence, I created the [`kaggle_titanic` module](https://github.com/chrischow/kaggle_titanic/blob/master/Modules/kaggle_titanic.py) to perform these three data cleaning functions, and many other functions for machine learning.  
  
### Model Tuning I - Improper Cross Validation
I created pipelines for performing feature selection for and tuning the various models:  
  
1. LightGBM RF with low depth
2. LightGBM RF with high depth
3. LightGBM GBM with low depth
4. LightGBM GBM with high depth
5. XGBoost GBM with low depth
6. XGBoost GBM with high depth
7. Logistic Regression
8. Ridge Regression
9. Linear Support Vector Classifier (SVC)
  
Having completed the tuning for the above models, I discovered another source of contamination: in the cross validation process. I was using cross validation to both (1) tune hyperparameters and (2) score models. These two processes should have been separate. The correct way to have performed cross validation was:  
  
1. Split data into **meta folds** (e.g. 5 repeats of 5 folds)
2. For each **meta fold**:
  a. Split training data into folds (e.g. 5 repeats of 5 folds)
  b. Run feature selection and hyperparameter tuning algorithms using cross validation
  c. Score the model on the test set of the **meta fold**
  d. Move on to next **meta fold** and repeat
3. Evaluate results for all **meta folds**
  
The first important thing I learned was that although feature selection and parameter tuning require cross validation, **all processes from data cleaning to feature generation to feature selection to hyperparameter tuning** require cross validation as well. In other words, **cross validation should be used to validate the full approach**. My mistake was that I saw hyperparameter tuning with cross validation as the end stage of testing. This was wrong. The second important thing I learned was that it was better to place feature selection before hyperparameter tuning. This is because changes in hyperparameter levels do not change how important or unimportant the features are. We only need the initial parameters for running RFE to not prevent the algorithm from letting the important features surface. For example, in tree-based models, increasing the number of trees, depth, or features sampled per tree would help to ensure that sufficient features are evaluated. The tradeoff to increasing these hyperparameters would be the computational cost. These two objectives should be balanced well.  

  
  
