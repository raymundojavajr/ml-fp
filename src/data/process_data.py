# src/data/process_data.py

from sklearn.preprocessing import LabelEncoder
from .load_data import load_data, find_root


def define_data_columns():
    """Return lists of categorical and numerical column names."""
    categorical_cols = ["Type", "Product ID", "Failure Type"]
    numerical_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    return categorical_cols, numerical_cols


def encode_categorical_columns(df, categorical_cols):
    """Encode categorical columns and return updated DataFrame and encoder dict."""
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict


def drop_original_categorical(df, categorical_cols):
    """Drop original categorical columns."""
    return df.drop(columns=categorical_cols)


def clean_column_names(df):
    """Clean column names by removing brackets and replacing spaces with underscores."""
    df.columns = df.columns.str.replace(r'[\[\]]', '', regex=True).str.replace(' ', '_')
    return df


def save_processed_data(df, relative_path="data/processed/predictive_maintenance_processed.csv"):
    """Save processed DataFrame to a CSV file."""
    root = find_root()
    save_path = root / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Processed data saved to: {save_path}")


if __name__ == "__main__":
    df = load_data()
    categorical_cols, numerical_cols = define_data_columns()
    df, _ = encode_categorical_columns(df, categorical_cols)
    df = drop_original_categorical(df, categorical_cols)
    df = clean_column_names(df)
    print("Processed DataFrame preview:")
    print(df.head())
    save_processed_data(df)
