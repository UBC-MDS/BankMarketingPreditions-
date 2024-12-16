import os
import pandas as pd


def clean_data(input_path, output_path):
    """
    Removes duplicate rows from a dataset.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the cleaned/transformed CSV file.

    Returns:
        None: Saves cleaned data to output_path.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The input file does not exist at: {input_path}")

    df = pd.read_csv(input_path, sep=";")
    
    # Remove duplicates
    df = df.drop_duplicates()

    df.to_csv(output_path, index=False)
if __name__ == "__main__":