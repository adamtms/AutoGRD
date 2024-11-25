from glob import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def PreProcessing(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)
    return X_processed

def CreateClassifications(path: str, output_path, seed=0):
    """Gets a dataset, and uses random forest to classify the instances. 
       Then, each instance is associated with some leaf. 
       These associations are written to file, which its location is output_path.

    Parameters
    ----------
    path : str
        The file location of the dataset(csv file)
    dataSet_name : str
        The dataset name
    output_path : str
        Path to the output file(txt file)
        

    Returns
    -------
    list
        a list of strings used that are the header columns
    """
    if path.endswith("csv"):
        df = pd.read_csv(path)
    elif path.endswith("dat"):
        df = pd.read_table(path)
    df = df.dropna(axis=1, how='all')
    features = df.columns[1: len(df.columns) -1]
    X = df[features] 
    target_col = df.columns[len(df.columns) -1]
    Y = df[target_col]
    X = PreProcessing(X)

    
    clf = RandomForestClassifier(max_depth=8, random_state=0, n_estimators=500, random_state=seed)
    clf.fit(X, Y)
    # For regression datasets, one sholud use RandomForestRegressor instead of RandomForestClassifier
    # from sklearn.ensemble import RandomForestRegressor
    # clf = RandomForestRegressor(max_depth=8, random_state=0, n_estimators=500)
    result = clf.apply(X)
    result = result.transpose()
    result = pd.DataFrame(result)
    
    result.to_csv(output_path, index=False, header=False)