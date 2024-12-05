import click
import os
import pandas as pd

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
    DataFrame
        Pandas DataFrame containing the contents of the CSV file.
    """
    file_path = os.path.join(directory, filename)


    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {filename} does not exist in the directory {directory}.")


    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read the CSV file from {file_path}")
        return df
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV file: {e}")

@click.command()
@click.option('--directory', type=str, default='/data', help="Directory where the CSV file is located.")
@click.option('--filename', type=str, default='bankmarketing.csv', help="Name of the CSV file.")
def main(directory, filename):
    """Reads a CSV file from a specified directory and prints its contents."""
    try:
      
        df = read_csv(directory, filename)

        
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
