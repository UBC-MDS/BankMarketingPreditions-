import pytest
import pandas as pd
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import clean_data


def test_clean_data(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    # Create a sample dataset with duplicates
    data = pd.DataFrame({
        'age': [25, 30, 25],
        'job': ["admin.", "technician", "admin."],
        'y': ["yes", "no", "yes"]
    })
    data.to_csv(input_path, sep=";", index=False)

    # Run the clean_data function
    clean_data(input_path, output_path)

    # Load the cleaned data
    cleaned_data = pd.read_csv(output_path, sep=";")

    # Check that duplicates were removed
    assert len(cleaned_data) == 2
    assert cleaned_data.duplicated().sum() == 0
