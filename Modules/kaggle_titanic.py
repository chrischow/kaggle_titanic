# * ---------------------------- *
# | KAGGLE TITANIC HELPER MODULE |
# * ---------------------------- *
# Import required modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE

# Settings
warnings.filterwarnings('ignore')
matplotlib.style.use('ggplot')

# Help function
def helpme():
    
    # Print
    print(
        """
        * ------------------------------- *
        | KAGGLE TITANIC HELPER FUNCTIONS |
        * ------------------------------- *
        
        # ---- INTRODUCTION ---- #
        The Kaggle Titanic module provides functions for end-to-end data mining, from data cleaning to feature generation
        to dataset splitting to model tuning.
        
        # ---- DATA CLEANING AND FEATURE GENERATION ---- #
        
        [ clean_all(df) ]
        
        Generates generic features for all data (train and test). This includes features for one-hot encoding (OH), ordinal
        categorical features (OC), binaries (BN), and numerics (NM). Numeric features in both the training and test set are
        not encoded at this stage.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ encode_train(df) ]
        
        Takes training data, performs mean encoding, and returns a cleaned training set, along with the mapping used.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ map_test(df, mapping) ]
        
        Takes test data and the mapping generated from encode_train, and outputs a cleaned test set.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ get_folds(df, random_state) ]
        
        Takes a dataset and splits it using RepeatedStratifiedKFold() with 5 repeats and 5 folds.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ split_train_test(df, folds, setno) ]
        
        Takes a set number (between 1 and 25), and splits data into training features, training labels, test features, and 
        test labels. Returns a tuple of length 4.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ prep_sets(df, folds, n_jobs = 3, sets = list(np.arange(1,26,1))) ]
        
        Splits data into a list of tuples, each of length 4. The training sets are specified using the "sets" argument.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ fit_score(estimator, sep_sets) ]
        
        Takes a model and a tuple of training and test features and labels, and returns a score.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ score_sets(estimator, all_sets, n_jobs = 3, verbose = False) ]
        
        Performs scoring on given pairs of train/test sets.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ score_grid_single(estimator, param, pval, all_sets, n_jobs = 3, verbose = False) ]
        
        Perform scoring on given train/test sets for a single parameter value.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ score_grid(estimator, params, all_sets, n_jobs = 3, verbose = False) ]
        
        Perform scoring on given train/test sets for a list of parameter values. Can only tune one parameter.
        
        -------------------------------------------------------------------------------------------------------------------
        
        [ custom_rfecv(estimator, nfeats, sub_sets, df, step = 1, n_jobs = 3, verbose = True) ]
        
        Perform RFE on all possible values (1 to total no. of features) for all given sets.
        
        -------------------------------------------------------------------------------------------------------------------
        """
    )

# Clean all data
def clean_all(input_df):
    
    # Copy data
    df = input_df.copy()
    
    # ---- PCLASS ---- #
    df['pclass_OH'] = 'CLASS' + df.Pclass.astype('str')
    df['pclass_OC'] = df.pclass_OH.astype('category', ordered = True, categories = ['CLASS3', 'CLASS2', 'CLASS1']).cat.codes

    # ---- TITLE ---- #
    df['title_OH'] = df.Name.str.replace('.*, ', '').str.replace(' .*', '')
    df.title_OH[(df.title_OH != 'Mr.') & (df.title_OH != 'Miss.') & (df.title_OH != 'Mrs.') & (df.title_OH != 'Master.')] = 'Other'

    # ---- SEX ---- #
    df['sex_BN'] = df.Sex.astype('category').cat.codes
    df['sex_OH'] = df.Sex.copy()

    # ---- SIBSP ---- #
    df['sibsp_OH'] = 'None'
    df['sibsp_OH'][df['SibSp'] == 1] = 'One'
    df['sibsp_OH'][df['SibSp'] >= 2] = 'Two or More'
    df['sibsp_NM'] = df['SibSp'].copy()
    df['sibsp_OC'] = df['sibsp_OH'].astype('category', ordered = True, categories = ['None', 'One', 'Two or More']).cat.codes

    # ---- PARCH ---- #
    df['parch_OH'] = 'None'
    df['parch_OH'][df['Parch'] == 1] = 'One'
    df['parch_OH'][df['Parch'] >= 2] = 'Two or More'
    df['parch_NM'] = df['Parch'].copy()
    df['parch_OC'] = df['parch_OH'].astype('category', ordered = True, categories = ['None', 'One', 'Two or More']).cat.codes

    # ---- TICKET NUMBER ---- #
    df['Ticket'] = df.Ticket.str.replace('[a-zA-z\\.]', '').str.replace('[\\/.* ]', '')
    df['ticlen_OH'] = 'L' + df['Ticket'].str.len().astype(str)
    df['ticlen_OH'][df['Ticket'].str.len().isin([0,1,3,7,8])] = 'LO'

    # ---- FARE ---- #
    df['fare_NM'] = df.Fare.copy()
    df['fare_NM'][df.Fare > 150] = 150
    df['fare_OC'] = pd.qcut(df.fare_NM, 6).astype('category').cat.codes
    df['fare_OH'] = 'G' + df.fare_OC.astype(str)

    # ---- CABIN LETTERS ---- #
    df['cab_letter'] = df.Cabin.str[0]
    df['cab_letter'][df.Cabin.isnull()] = 'NIL'
    df['cabletter_OH'] = 'B/D/E'
    df['cabletter_OH'][df.cab_letter.isin(['A', 'G', 'T', 'C', 'F'])] = 'Others'
    df['cabletter_OH'][df.cab_letter.isin(['NIL'])] = 'NIL'

    # ---- CABIN NUMBERS ---- #
    temp_df = df.Cabin.str.replace('[a-zA-z]', '').str.split(' ', expand = True)
    temp_df.columns = ['a','b','c','d']
    temp_df.a = pd.to_numeric(temp_df.a, errors = 'coerce')
    temp_df.b = pd.to_numeric(temp_df.b, errors = 'coerce')
    temp_df.c = pd.to_numeric(temp_df.c, errors = 'coerce')
    temp_df.d = pd.to_numeric(temp_df.d, errors = 'coerce')
    df['cab_no'] = temp_df.mean(axis = 1)
    df['cabno_OH'] = 'C1'
    df['cabno_OH'][(df.cab_no > 35) & (df.cab_no <= 49)] = 'C2'
    df['cabno_OH'][(df.cab_no > 70.143) & (df.cab_no <= 96.143)] = 'C2'
    df['cabno_OH'][df.cab_no.isnull()] = 'NIL'

    # ---- EMBARKED ---- #
    df['embarked_OH'] = df.Embarked.copy()
    df['embarked_OH'][df['embarked_OH'].isnull()] = 'C'

    # ---- FINAL CLEANING ---- #
    # Drop variables
    df = df.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'cab_letter', 'cab_no', 'Cabin', 'Embarked'], axis = 1)
    
    # Output
    return df

# Encode training data
def encode_train(input_df):
    
    # Copy data
    df = input_df.copy()
    
    # Initialise mappings
    mapping = dict()
    
    # ---- AGE ---- #
    # Set seed
    seed = 123

    # Mask for missing ages
    miss_age = df.Age.isnull()

    # Masks for each title
    miss_master = df.title_OH == 'Master.'
    miss_ms = df.title_OH == 'Miss.'
    miss_mr = df.title_OH == 'Mr.'
    miss_mrs = df.title_OH == 'Mrs.'
    miss_other = df.title_OH == 'Other'

    # Impute ages
    np.random.seed(seed)
    df['Age'][miss_age & miss_master] = np.random.normal(
        df.groupby('title_OH').Age.median()[0],
        df.groupby('title_OH').Age.std()[0] * 0.5,
        df['Age'][miss_age & miss_master].shape[0]
    )

    df['Age'][miss_age & miss_ms] = np.random.normal(
        df.groupby('title_OH').Age.median()[1],
        df.groupby('title_OH').Age.std()[1] * 0.5,
        df['Age'][miss_age & miss_ms].shape[0]
    )

    df['Age'][miss_age & miss_mr] = np.random.normal(
        df.groupby('title_OH').Age.median()[2],
        df.groupby('title_OH').Age.std()[2] * 0.5,
        df['Age'][miss_age & miss_mr].shape[0]
    )

    df['Age'][miss_age & miss_mrs] = np.random.normal(
        df.groupby('title_OH').Age.median()[3],
        df.groupby('title_OH').Age.std()[3] * 0.5,
        df['Age'][miss_age & miss_mrs].shape[0]
    )

    df['Age'][miss_age & miss_other] = np.random.normal(
        df.groupby('title_OH').Age.median()[4],
        df.groupby('title_OH').Age.std()[4] * 0.5,
        df['Age'][miss_age & miss_other].shape[0]
    )
    
    # Add to mapping
    mapping['age_master'] = (df.groupby('title_OH').Age.median()[0], df.groupby('title_OH').Age.std()[0] * 0.5)
    mapping['age_ms'] = (df.groupby('title_OH').Age.median()[1], df.groupby('title_OH').Age.std()[1] * 0.5)
    mapping['age_mr'] = (df.groupby('title_OH').Age.median()[2], df.groupby('title_OH').Age.std()[2] * 0.5)
    mapping['age_mrs'] = (df.groupby('title_OH').Age.median()[3], df.groupby('title_OH').Age.std()[3] * 0.5)
    mapping['age_other'] = (df.groupby('title_OH').Age.median()[4], df.groupby('title_OH').Age.std()[4] * 0.5)
    
    # Split into 10 quantiles
    df['age_OC'] = pd.qcut(df.Age, 10).astype('category').cat.codes
    _, mapping['age_bins'] = pd.qcut(df.Age, 10, retbins = True)

    # Age features
    df['age_NM'] = df.Age
    df['age_OH'] = 'A' + df.age_OC.astype(str)
    
    # ---- PCLASS ---- #
    map_pclass = df.groupby('pclass_OH').Survived.mean()
    mapping['pclass_ME'] = map_pclass
    df['pclass_ME'] = df.pclass_OH.map(map_pclass)
    
    # ---- TITLE ---- #
    map_title = df.groupby('title_OH').Survived.mean()
    mapping['title_ME'] = map_title
    df['title_ME'] = df.title_OH.map(map_title)
    
    # ---- AGE ---- #
    map_age = df.groupby('age_OH').Survived.mean()
    mapping['age_ME'] = map_age
    df['age_ME'] = df.age_OH.map(map_age)
    
    # ---- SIBSP ---- #
    map_sibsp = df.groupby('sibsp_OH').Survived.mean()
    mapping['sibsp_ME'] = map_sibsp
    df['sibsp_ME'] = df.sibsp_OH.map(map_sibsp)
    
    # ---- PARCH ---- #
    map_parch = df.groupby('parch_OH').Survived.mean()
    mapping['parch_ME'] = map_parch
    df['parch_ME'] = df.parch_OH.map(map_parch)
    
    # ---- TICKET ---- #
    map_ticlen = df.groupby('ticlen_OH').Survived.mean()
    mapping['ticlen_ME'] = map_ticlen
    df['ticlen_ME'] = df['ticlen_OH'].map(map_ticlen)
    
    # ---- FARE ---- #
    map_fare = df.groupby('fare_OH').Survived.mean()
    mapping['fare_ME'] = map_fare
    df['fare_ME'] = df['fare_OH'].map(map_fare)
    
    # ---- CAB LETTER ---- #
    map_cabletter = df.groupby('cabletter_OH').Survived.mean()
    mapping['cabletter_ME'] = map_cabletter
    df['cabletter_ME'] = df['cabletter_OH'].map(map_cabletter)
    
    # ---- CAB NUMBER ---- #
    map_cabno = df.groupby('cabno_OH').Survived.mean()
    mapping['cabno_ME'] = map_cabno
    df['cabno_ME'] = df['cabno_OH'].map(map_cabno)
    
    # ---- EMBARKED ---- #
    map_embarked = df.groupby('embarked_OH').Survived.mean()
    mapping['embarked_ME'] = map_embarked
    df['embarked_ME'] = df['embarked_OH'].map(map_embarked)
    
    # Drop age
    df = df.drop('Age', axis = 1)
    
    # Output
    output = (df, mapping)
    
    return output

# Map to test data
def map_test(input_df, mapping):
    
    # Copy data
    df = input_df.copy()
    
    # All mapping
    # ---- AGE ---- #
    # Set seed
    seed = 123

    # Mask for missing ages
    miss_age = df.Age.isnull()

    # Masks for each title
    miss_master = df.title_OH == 'Master.'
    miss_ms = df.title_OH == 'Miss.'
    miss_mr = df.title_OH == 'Mr.'
    miss_mrs = df.title_OH == 'Mrs.'
    miss_other = df.title_OH == 'Other'

    # Impute ages
    np.random.seed(seed)
    df['Age'][miss_age & miss_master] = np.random.normal(
        mapping['age_master'][0],
        mapping['age_master'][1] * 0.5,
        df['Age'][miss_age & miss_master].shape[0]
    )

    df['Age'][miss_age & miss_ms] = np.random.normal(
        mapping['age_ms'][0],
        mapping['age_ms'][1] * 0.5,
        df['Age'][miss_age & miss_ms].shape[0]
    )

    df['Age'][miss_age & miss_mr] = np.random.normal(
        mapping['age_mr'][0],
        mapping['age_mr'][1] * 0.5,
        df['Age'][miss_age & miss_mr].shape[0]
    )

    df['Age'][miss_age & miss_mrs] = np.random.normal(
        mapping['age_mrs'][0],
        mapping['age_mrs'][1] * 0.5,
        df['Age'][miss_age & miss_mrs].shape[0]
    )

    df['Age'][miss_age & miss_other] = np.random.normal(
        mapping['age_other'][0],
        mapping['age_other'][1] * 0.5,
        df['Age'][miss_age & miss_other].shape[0]
    )
    
    # Split into 10 quantiles
    bins = mapping['age_bins']
    df['age_OC'] = 0
    df['age_OC'][(df.Age > bins[1]) & (df.Age <= bins[2])] = 1
    df['age_OC'][(df.Age > bins[2]) & (df.Age <= bins[3])] = 2
    df['age_OC'][(df.Age > bins[3]) & (df.Age <= bins[4])] = 3
    df['age_OC'][(df.Age > bins[4]) & (df.Age <= bins[5])] = 4
    df['age_OC'][(df.Age > bins[5]) & (df.Age <= bins[6])] = 5
    df['age_OC'][(df.Age > bins[6]) & (df.Age <= bins[7])] = 6
    df['age_OC'][(df.Age > bins[7]) & (df.Age <= bins[8])] = 7
    df['age_OC'][(df.Age > bins[8]) & (df.Age <= bins[9])] = 8
    df['age_OC'][(df.Age > bins[9]) & (df.Age <= bins[10])] = 9

    # Age features
    df['age_NM'] = df.Age
    df['age_OH'] = 'A' + df.age_OC.astype(str)
    df['age_ME'] = df.age_OH.map(mapping['age_ME'])
    
    # Remove age
    df = df.drop('Age', axis = 1)
    
    # ---- PCLASS ---- #
    df['pclass_ME'] = df.pclass_OH.map(mapping['pclass_ME'])
    
    # ---- TITLE ---- #
    df['title_ME'] = df.title_OH.map(mapping['title_ME'])
    
    # ---- SIBSP ---- #
    df['sibsp_ME'] = df.sibsp_OH.map(mapping['sibsp_ME'])
    
    # ---- PARCH ---- #
    df['parch_ME'] = df.parch_OH.map(mapping['parch_ME'])
    
    # ---- TICKET ---- #
    df['ticlen_ME'] = df['ticlen_OH'].map(mapping['ticlen_ME'])
    
    # ---- FARE ---- #
    df['fare_ME'] = df['fare_OH'].map(mapping['fare_ME'])
    
    # ---- CAB LETTER ---- #
    df['cabletter_ME'] = df['cabletter_OH'].map(mapping['cabletter_ME'])
    
    # ---- CAB NUMBER ---- #
    df['cabno_ME'] = df['cabno_OH'].map(mapping['cabno_ME'])
    
    # ---- EMBARKED ---- #
    df['embarked_ME'] = df['embarked_OH'].map(mapping['embarked_ME'])
    
    # Output
    return df

# Get indices for Repeated Stratified K Folds
def get_folds(df, random_state):
    
    # Initialise output
    output = dict()

    # Initialise counter
    counter = 1
    
    # Create labels
    y = df['Survived']
    
    # Separate features
    X = df.drop('Survived', axis = 1)
    
    # Create
    rkf = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=random_state)
    
    # Loop through folds
    for tr_index, te_index in rkf.split(X, y):

        # Append to dictionary
        output['set' + str(counter)] = dict()
        output['set' + str(counter)]['train'] = tr_index
        output['set' + str(counter)]['test'] = te_index
        
        # Update counter
        counter += 1
    
    # Return
    return output

# Split data into X_train, y_train, X_test, y_test
def split_train_test(df, folds, setno, feats = False):
    
    # Access set number
    temp_indices = folds['set' + str(setno)]
    
    # Configure train and test data
    train = df.iloc[temp_indices['train'], :]
    test = df.iloc[temp_indices['test'], :]
    
    # Encode train data
    train_encoded, temp_map = encode_train(train)
    
    # Map test data
    test_encoded = map_test(test, temp_map)
    
    # Separate features from target
    X_train = train_encoded.drop('Survived', axis = 1)
    X_train = pd.get_dummies(X_train)
    y_train = train_encoded['Survived']
    
    X_test = test_encoded.drop('Survived', axis = 1)
    X_test = pd.get_dummies(X_test)
    y_test = test_encoded['Survived']
    
    # Check for non-overlapping columns
    rem_col = list(set(X_train.columns) - set(X_test.columns))
    
    # Remove from sets
    if X_train.columns.isin(rem_col).any():
        
        X_train = X_train.drop(rem_col, axis = 1)
    
    if X_test.columns.isin(rem_col).any():
        
        X_test = X_test.drop(rem_col, axis = 1)
    
    # Remove features
    if feats:
        
        selected_feats = [f for f in feats if np.sum(X_train.columns.isin([f]).any()) >= 1]
        X_train = X_train[selected_feats]
        X_test = X_test[selected_feats]
    
    # Output
    return (X_train, y_train, X_test, y_test)

# Get list of sets
def prep_sets(df, folds, n_jobs = 3, sets = list(np.arange(1,26,1)), feats = False):
    
    # Get all sets
    output = Parallel(n_jobs=n_jobs, verbose = False)(delayed(split_train_test)(df=df, folds=folds, setno=s,
                                                                               feats = feats) for s in sets)
    
    # Output
    return output

# Fit and predict
def fit_score(estimator, sep_sets):
    
    # Fit model
    estimator.fit(sep_sets[0], sep_sets[1])
    
    # Score model
    output = estimator.score(sep_sets[2], sep_sets[3])
    
    return output

# Perform scoring on all provided sets
def score_sets(estimator, all_sets, n_jobs = 3, verbose = False):
    
    # Run
    output = Parallel(n_jobs=n_jobs, verbose = verbose)(delayed(fit_score)(estimator, d) for d in all_sets)
    
    return output

# Score model
def score_grid_single(estimator, param, pval, all_sets, n_jobs = 3, verbose = False):
    
    # Set parameter values
    model = estimator.set_params(**{param: pval})
    
    # Compute scores
    temp_scores = score_sets(
        estimator = model,
        all_sets = all_sets,
        n_jobs = n_jobs,
        verbose = verbose
    )
    
    # Return mean
    return np.mean(temp_scores)

# Single parameter grid search
def score_grid(estimator, params, all_sets, n_jobs = 3, verbose = False):
    
    # Output lists
    param = list(params.keys())[0]
    param_value = list(params.values())[0]
    mean_test_score = []
    
    # Compute mean test scores
    for pval in [v for k, v in params.items()][0]:
        
        # Update
        if verbose:
            print('Computing test scores for ' + param + ' = ' + str(pval) + '...', end = "", flush = True)
        
        # Compute score
        temp_score = score_grid_single(
                estimator = estimator,
                param = param,
                pval = pval,
                all_sets = all_sets,
                n_jobs = n_jobs
        )
        
        # Append
        mean_test_score.append(temp_score)
        
        # Update
        print('Done!')
    
    # Spacing
    print()
    print()
    
    # Print results
    max_score = np.max(mean_test_score)
    opt_value = param_value[np.argmax(mean_test_score)]
    print('[ RESULTS ]')
    print('   Best Score: ' + str(max_score))
    print('Optimal Value: ' + str(opt_value))
    
    # Plot
    plt.plot(param_value, mean_test_score)
    plt.title(param)
    plt.show()
    
    # Output
    output = pd.DataFrame([param_value, mean_test_score], index = ['param_value','mean_test_score']).T
    output.insert(0, "param", param)
    return (output, opt_value)

def rfe_single(estimator, dat, n_jobs = 3, xg = False, verbose = False):
    
    # Set up RFE
    rfe = RFE(
        estimator = estimator,
        n_features_to_select = 1,
        step = 1,
        verbose = verbose
    )
    
    # Extract data
    X_train = dat[0]
    y_train = dat[1]
    X_test = dat[2]
    y_test = dat[3]
    
    # Set up scorer
    if xg:
        temp_scorer = lambda est, features: accuracy_score(y_true=y_test, y_pred=est.predict(X_test.values[:, features]))
    
    else:
        temp_scorer = lambda est, features: accuracy_score(y_true=y_test, y_pred=est.predict(X_test.iloc[:, features]))
    
    # Fit RFE
    rfe._fit(X_train, y_train, temp_scorer)
    
    # Return scores
    return rfe.scores_

# Function to perform RFECV
def custom_rfecv(estimator, sub_sets, df, step = 1, n_jobs = 3, xg = False, verbose = True):
    
    # Set parameters
    n_features = int(stats.mode([ sub_sets[i][0].shape[1] for i in range(len(sub_sets)) ], axis = None)[0])
    
    # Compute scores
    all_scores = Parallel(n_jobs=n_jobs, verbose = verbose)(delayed(rfe_single)(
        estimator=estimator, dat=d, n_jobs=n_jobs, xg = xg, verbose = False
    ) for d in sub_sets)
    
    # Check number of minimum entries
    score_lengths = [len(i) for i in all_scores]
    common = int(stats.mode(score_lengths, axis = None)[0])
    rem_idx = [i for i in range(len(score_lengths)) if score_lengths[i] != common]
    for idx in sorted(rem_idx, reverse=True):
        del all_scores[idx]
    
    # Consolidate scores
    all_scores = np.mean(all_scores, axis=0)
    n_features_to_select = max(n_features - (np.argmax(all_scores) * step), 1)
    
    # Print results
    max_score = np.max(all_scores)
    print('   Best Score: ' + str(max_score))
    print('Optimal Value: ' + str(n_features_to_select))
    
    # Estimate model
    opt_rfe = RFE(
            estimator = estimator,
            step = step,
            n_features_to_select = n_features_to_select
    )
    
    # Extract data
    X_final = pd.get_dummies(df.drop('Survived', axis = 1))
    y_final = df.Survived
    
    opt_rfe.fit(X_final, y_final)
    
    # Extract features
    opt_feats = X_final.columns[opt_rfe.ranking_ == 1]
    
    # Configure X-axis for plot
    feat_nums = list(np.arange(1, n_features+1, 1))
    
    # Plot
    plt.plot(feat_nums, list(reversed(all_scores)))
    plt.title("Mean Test Score vs. No. of Features")
    plt.show()
    
    # Output
    output = pd.DataFrame([n_features, all_scores], index = ['n_features','mean_test_score']).T
    return (output, n_features_to_select, opt_feats)