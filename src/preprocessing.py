import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(path, sep=';')

def preprocess_data(df: pd.DataFrame):
    """
    Encode categorical features and the target variable.
    """
    df = df.copy()
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    X = pd.get_dummies(df.drop('y', axis=1), drop_first=True)
    y = df['y']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
