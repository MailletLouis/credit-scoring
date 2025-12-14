import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Charge les fichiers CSV"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def clean_data(df):
    """Nettoyage des données : doublons, valeurs manquantes"""
    df = df.drop_duplicates()
    # Exemple simple : remplir les NaN par la médiane
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Unknown', inplace=True)
    return df

def encode_categorical(df):
    """Encodage des variables catégorielles avec get_dummies"""
    df = pd.get_dummies(df, drop_first=True)
    return df

def create_features(df):
    """Création de nouvelles features si nécessaire"""
    # Exemple : ratio crédit / revenu
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    return df

def train_test_split_data(df, target_col='TARGET', test_size=0.2, random_state=42):
    """Split train/test"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

