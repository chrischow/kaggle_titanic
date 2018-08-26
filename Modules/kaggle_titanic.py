# * ---------------------------- *
# | KAGGLE TITANIC HELPER MODULE |
# * ---------------------------- *
# Import required modules
import pandas as pd
import numpy as np

# Clean all data
def clean_all(df):
    
    # ---- PCLASS ---- #
    df['pclass_OH'] = 'CLASS' + df.Pclass.astype('str')
    df['pclass_OC'] = df.pclass_OH.astype('category', ordered = True, categories = ['CLASS3', 'CLASS2', 'CLASS1']).cat.codes

    # ---- TITLE ---- #
    df['title_OH'] = df.Name.str.replace('.*, ', '').str.replace(' .*', '')
    df.title_OH[(df.title_OH != 'Mr.') & (df.title_OH != 'Miss.') & (df.title_OH != 'Mrs.') & (df.title_OH != 'Master.')] = 'Other'

    # ---- SEX ---- #
    df['sex_BN'] = df.Sex.astype('category').cat.codes
    df['sex_OH'] = df.Sex.copy()

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

    # Split into 10 quantiles
    df['age_OC'] = pd.qcut(df.Age, 10).astype('category').cat.codes

    # Age features
    df['age_NM'] = df.Age
    df['age_OH'] = 'A' + df.age_OC.astype(str)

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
    df = df.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'cab_letter', 'cab_no', 'Cabin', 'Embarked'], axis = 1)
    
    # Output
    return df

# Encode training data
def encode_train(df):
    
    # Initialise mappings
    mapping = dict()
    
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
    
    # Output
    output = {
        'data': df,
        'mapping': mapping
    }
    
    return output

# Map test data
def map_test(df, mapping):
    
    # All mapping
    # ---- PCLASS ---- #
    df['pclass_ME'] = df.pclass_OH.map(mapping['pclass_ME'])
    
    # ---- TITLE ---- #
    df['title_ME'] = df.title_OH.map(mapping['title_ME'])
    
    # ---- AGE ---- #
    df['age_ME'] = df.age_OH.map(mapping['age_ME'])
    
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