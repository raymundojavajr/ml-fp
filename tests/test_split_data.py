import pytest
import pandas as pd
from src.data.split_data import split_data

def test_split_data_success():
    # Create a sample DataFrame
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'Target': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Call the split_data function
    X_train, X_test, y_train, y_test = split_data(df, target='Target', test_size=0.2, random_state=42)

    # Check the shapes of the splits: with 5 rows and test_size=0.2, expect 1 row for test and 4 for train.
    assert X_train.shape == (4, 2)
    assert X_test.shape == (1, 2)
    assert y_train.shape == (4,)
    assert y_test.shape == (1,)

def test_split_data_key_error():
    # Create a sample DataFrame without the target column
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(data)

    # Expect a ValueError when the target column is missing
    with pytest.raises(ValueError, match="Target column 'Target' not found in data"):
        split_data(df, target='Target', test_size=0.2, random_state=42)

def test_split_data_with_invalid_data():
    # Create a sample DataFrame with "invalid" data (non-numeric value in 'feature1')
    data = {
        'feature1': [1, 2, 3, 4, 'invalid'],
        'feature2': [5, 4, 3, 2, 1],
        'Target': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # Call the split_data function; it should complete without raising an error.
    X_train, X_test, y_train, y_test = split_data(df, target='Target', test_size=0.2, random_state=42)

    # Check that the total number of rows in X_train and X_test equals the number of rows in df.
    total_rows = X_train.shape[0] + X_test.shape[0]
    assert total_rows == len(df)
