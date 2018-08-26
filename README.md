# Kaggle Competition - Titanic Dataset
This Kaggle competition requires participants to predict whether passengers will survive.

## Notebooks
The notebooks are named in chronological order. First, I performed exploratory data analysis (EDA) on all features in the dataset. Concurrently, I cleaned the data and generated new features. Second, I experimented with a Random Forest model (`sklearn` implementation). By the final iteration in the optimisation process, the accuracy was **82.47%**. However, the `sklearn` implementation (`RandomForestClassifier`) is extremely slow. Hence, I switched over to the `lightgbm` implementation of the Random Forest algorithm. It was much faster. The subsequent three notebooks capture my attempts at figuring out the best sequence for the tuning of parameters. The best approach was to:  
  
1. Tune the number of trees/estimators
2. Perform Recursive Feature Elimination (RFE) with cross-validation (CV)
3. Tune the proportion of observations sampled
4. Tune the maximum features used per split
5. Tune the minimum samples in each terminal node
6. Tune alpha, the L1 regularisation parameter
7. Tune lambda, the L1 regularisation parameter
  
This tuning sequence resulted in a 5-fold CV test score of **83.46%**.  
  
### Regenerating Features
Then, I realised that I made a mistake in the feature generation process. I had generated encoded features using the entire training set. This meant that the pseudo-test sets in CV contained information about the target, and this could have inflated the test scores. Instead, I should have (1) performed data cleaning on the competition training and test sets together, (2) encoded the training set while saving the mappings, and (3) performed mappings on the test set **for each iteration of CV**.  
  
Hence, I created the `kaggle_titanic` module (in Modules folder) to perform these three functions.
