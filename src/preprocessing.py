import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(file_path):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def data_info(df):
    """
    Display basic information about dataset.
    """
    print("=" * 50)
    print("Dataset Shape :", df.shape)
    print("=" * 50)
    print("\nDataset Info:\n")
    print(df.info())
    print("\nMissing Values:\n")
    print(df.isnull().sum())


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    print(f"\nRemoved {before - after} duplicate rows.")
    return df


def handle_missing_values(df):
    """
    Fill missing values using Mean strategy.
    """
    imputer = SimpleImputer(strategy="mean")

    df[:] = imputer.fit_transform(df)

    return df


def split_features_target(df, target_column="TenYearCHD"):
    """
    Split dataset into Features (X) and Target (y)
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_data(X_train, X_test):
    """
    Standardize numerical features.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_test_scaled


def preprocess_data(file_path):
    """
    Complete preprocessing pipeline.
    """

    print("\nLoading Dataset...")
    df = load_data(file_path)

    print("\nChecking Dataset...")
    data_info(df)

    print("\nRemoving Duplicates...")
    df = remove_duplicates(df)

    print("\nHandling Missing Values...")
    df = handle_missing_values(df)

    print("\nSplitting Features and Target...")
    X, y = split_features_target(df)

    print("\nTrain Test Split...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # -----------------------------
    # Save Train and Test CSV
    # -----------------------------
    train_df = X_train.copy()
    train_df["TenYearCHD"] = y_train.values
    train_df.to_csv("data/train.csv", index=False)
    
    test_df = X_test.copy()
    test_df["TenYearCHD"] = y_test.values
    test_df.to_csv("data/test.csv", index=False)
    
    print("✔ train.csv Saved")
    print("✔ test.csv Saved")
    
    print("\nScaling Features...")
    X_train_scaled, X_test_scaled = scale_data(
        X_train,
        X_test
    )

    print("\nPreprocessing Completed Successfully!")

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        df
    )