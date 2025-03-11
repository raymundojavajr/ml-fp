import pytest
import pandas as pd
from unittest.mock import patch
from src.data.load_data import find_root, load_data, load_processed_data

def test_find_root_success(tmp_path):
    # Create a temporary README.md file in the temporary directory
    (tmp_path / "README.md").touch()
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        assert find_root() == tmp_path

def test_find_root_failure(tmp_path):
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(FileNotFoundError):
            find_root()

@patch("src.data.load_data.find_root")
@patch("pandas.read_csv")
def test_load_data_success(mock_read_csv, mock_find_root, tmp_path):
    # Arrange: Set up a temporary directory with the expected CSV file.
    mock_find_root.return_value = tmp_path
    dummy_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_csv.return_value = dummy_df
    (tmp_path / "data/raw").mkdir(parents=True, exist_ok=True)
    # Create an empty file to simulate file existence.
    (tmp_path / "data/raw/predictive_maintenance.csv").touch()

    # Act: Call load_data.
    df = load_data()

    # Assert: Verify the read_csv was called with the correct path and a non-empty DataFrame is returned.
    mock_read_csv.assert_called_once_with(tmp_path / "data/raw/predictive_maintenance.csv")
    assert not df.empty

@patch("src.data.load_data.find_root")
def test_load_data_file_not_found(mock_find_root, tmp_path):
    # Arrange: Ensure the file does not exist.
    mock_find_root.return_value = tmp_path
    # Act & Assert: load_data should raise a RuntimeError if the file is not found.
    with pytest.raises(RuntimeError, match="Data file not found"):
        load_data()

@patch("src.data.load_data.find_root")
@patch("pandas.read_csv")
def test_load_processed_data_success(mock_read_csv, mock_find_root, tmp_path):
    # Arrange: Set up a temporary directory with the expected processed CSV file.
    mock_find_root.return_value = tmp_path
    dummy_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_csv.return_value = dummy_df
    (tmp_path / "data/processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/processed/predictive_maintenance_processed.csv").touch()

    # Act: Call load_processed_data.
    df = load_processed_data()

    # Assert: Verify the read_csv was called with the correct path and a non-empty DataFrame is returned.
    mock_read_csv.assert_called_once_with(tmp_path / "data/processed/predictive_maintenance_processed.csv")
    assert not df.empty

@patch("src.data.load_data.find_root")
def test_load_processed_data_file_not_found(mock_find_root, tmp_path):
    # Arrange: Ensure the processed file does not exist.
    mock_find_root.return_value = tmp_path
    # Act & Assert: load_processed_data should raise a RuntimeError if the file is not found.
    with pytest.raises(RuntimeError, match="Processed data file not found"):
        load_processed_data()
