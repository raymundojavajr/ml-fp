# src/split_data.py

from sklearn.model_selection import train_test_split

def split_data(df, target, test_size=0.2, random_state=42):
    """Splits the DataFrame into training and test sets based on the target column."""
    # Separate features and target variable
    X = df.drop(columns=[target])
    y = df[target]
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Test block to verify functionality when running this module directly
if __name__ == "__main__":
    from .load_data import load_data  # Relative import from the same package
    # Load the data using the load_data function
    df = load_data()
    # Split the data using "Target" as the target column
    X_train, X_test, y_train, y_test = split_data(df, target="Target")
    # Print out shapes of the split data for verification
    print("Training features shape:", X_train.shape)
    print("Test features shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Test target shape:", y_test.shape)
