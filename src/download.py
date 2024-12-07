import os
import pandas as pd
import click


def read_csv(directory, filename):
    """
    Read a CSV file from the given specified directory.

    Parameters:
    ----------
    directory : str
        The directory where the CSV file is located.
    filename : str
        The name of the CSV file to be read.

    Returns:
    -------
    DataFrame, str
        The loaded DataFrame and the full file path.
    """
    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {filename} does not exist in the directory {directory}.")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read the CSV file from {file_path}")
        return df, file_path
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV file: {e}")


@click.command()
@click.option('--directory', type=str, default='data/bankmarketing/bank-additional/bank-additional/', help="Directory where the CSV file is located.")
@click.option('--filename', type=str, default='bank-additional-full.csv', help="Name of the CSV file.")
def main(directory, filename):
    """
    Reads a CSV file from a specified directory and prints its contents.
    Also outputs the file path for use in other scripts.
    """
    try:
        _, file_path = read_csv(directory, filename)

        # Output the file path for the next script
        print(f"File saved to: {file_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
