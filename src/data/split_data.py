# src/data/split_data.py

from sklearn.model_selection import train_test_split


def split_data(df, target, test_size=0.2, random_state=42):
    """Split DataFrame into train and test sets using target column."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    from .load_data import load_data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, target="Target")
    print("Training features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Test target shape:", y_test.shape)
