import os
import pandas as pd
import click
import pandera as pa
from pandera import Column, DataFrameSchema
import matplotlib.pyplot as plt

@click.command()
@click.option('--input_path', type=str, required=True, help="Path to the input CSV file.")
@click.option('--output_path', type=str, required=True, help="Path to save the cleaned/transformed CSV file.")
def preprocess_data(input_path, output_path):
    """
    Reads data from the specified input path, validates it, and performs preprocessing.
    """
    try:
        # Step 1: Load the data with correct delimiter
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"The input file does not exist at: {input_path}")
        print(f"Reading data from: {input_path}")
        bank_data = pd.read_csv(input_path, sep=";")  # Use the correct delimiter

        # Step 2: Clean and fix column names
        print("Current column names:", bank_data.columns.tolist())
        bank_data.columns = bank_data.columns.str.strip().str.replace('"', "")
        expected_columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 
                            'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 
                            'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 
                            'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
        if set(expected_columns).difference(bank_data.columns):
            raise ValueError(f"Incorrect column names. Expected columns: {expected_columns}")

        # Step 3: Remove duplicates
        if bank_data.duplicated().any():
            print(f"Found {bank_data.duplicated().sum()} duplicate rows. Removing duplicates.")
            bank_data = bank_data.drop_duplicates()

        # Step 4: Validate data
        validation_errors = validate_data(bank_data)
        if validation_errors:
            print("Data validation failed with the following errors:")
            for error in validation_errors:
                print(f"- {error}")
            return  # Exit if validation fails

        print("Data validation passed!")

        # Step 5: Calculate correlations with the target
        correlations = check_correlations_with_target(bank_data)  # Ensure this is called only once
        print("\nCorrelations with target variable:\n", correlations)

        # Step 6: Save cleaned data
        bank_data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")

        
def validate_data(df):
    errors = []

    # 1. Correct column names
    expected_columns = [
        'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
        'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'
    ]
    if not set(expected_columns).issubset(df.columns):
        errors.append(f"Incorrect column names. Expected columns: {expected_columns}")

    # 2. No empty observations
    if df.isnull().all(axis=1).any():
        errors.append("Dataset contains rows with all empty values.")

    # 3. Missingness not beyond expected threshold
    threshold = 0.1  # 10% threshold for missing data
    missing_ratios = df.isnull().mean()
    if (missing_ratios > threshold).any():
        high_missing_cols = missing_ratios[missing_ratios > threshold].index.tolist()
        errors.append(f"Columns with missingness beyond {threshold * 100}%: {high_missing_cols}")

    # 4. No duplicate observations
    if df.duplicated().any():
        errors.append("Dataset contains duplicate rows.")

    return errors


def check_outliers(df):
    """Checks for outliers in numeric columns."""
    print("Running outlier validation...")
    outlier_schema = DataFrameSchema(
        {
            "age": Column(pa.Int, pa.Check(lambda x: 17 <= x <= 100, name="age_check")),
            "duration": Column(pa.Int, pa.Check(lambda x: x >= 0, name="duration_check")),
            "campaign": Column(pa.Int, pa.Check(lambda x: x >= 0, name="campaign_check")),
            "pdays": Column(pa.Int, pa.Check(lambda x: x >= -1, name="pdays_check")),
            "previous": Column(pa.Int, pa.Check(lambda x: x >= 0, name="previous_check")),
            "emp.var.rate": Column(pa.Float, pa.Check(lambda x: -3.5 <= x <= 3, name="emp_var_rate_check")),
            "cons.price.idx": Column(pa.Float, pa.Check(lambda x: 92 <= x <= 95, name="cons_price_idx_check")),
            "cons.conf.idx": Column(pa.Float, pa.Check(lambda x: -51 <= x <= 50, name="cons_conf_idx_check")),
            "euribor3m": Column(pa.Float, pa.Check(lambda x: 0 <= x <= 6, name="euribor3m_check")),
            "nr.employed": Column(pa.Float, pa.Check(lambda x: 4900 <= x <= 5500, name="nr_employed_check")),
        }
    )
    try:
        outlier_schema.validate(df)
        print("Outlier validation passed!")
    except pa.errors.SchemaError as e:
        raise ValueError(f"Outlier validation failed:\n{e}")


def validate_categories(df):
    """Validates categorical column values."""
    print("Validating category levels...")
    expected_categories = {
        "job": ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", 
                "student", "blue-collar", "self-employed", "retired", "technician", "services"],
        "marital": ["married", "divorced", "single", "unknown"],
        "education": ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course',
                      'unknown', 'university.degree', 'illiterate'],
        "default": ["yes", "no", "unknown"],
        "housing": ["yes", "no", "unknown"],
        "loan": ["yes", "no", "unknown"],
        "contact": ["unknown", "telephone", "cellular"],
        "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
        "poutcome": ['nonexistent', 'failure', 'success']
    }

    for col, categories in expected_categories.items():
        if not set(df[col].unique()).issubset(categories):
            raise ValueError(f"Column '{col}' has unexpected category levels.")
    print("Category validation passed!")


def validate_target(df):
    """Validates the target variable."""
    print("Validating target variable...")
    target_schema = pa.DataFrameSchema({
        "y": pa.Column(str, pa.Check.isin(['yes', 'no'], error="Target must be 'yes' or 'no'"), nullable=False)
    })
    try:
        target_schema.validate(df)
        print("Target validation passed!")
    except pa.errors.SchemaError as e:
        print(f"Target validation failed: {e}")


def check_correlations_with_target(df):
    """Checks correlations with the target variable."""
    print("Checking correlations with the target variable...")
    
    # Ensure the target variable is numeric for correlation calculation
    encoded_df = df.copy()
    for col in encoded_df.select_dtypes(include=['object']).columns:
        encoded_df[col] = encoded_df[col].astype('category').cat.codes

    if 'y' not in encoded_df.columns:
        raise ValueError("Target variable 'y' is not found in the dataset.")

    # Calculate correlation with the target variable
    correlations = encoded_df.corr()["y"].drop("y")  # Drop self-correlation
    # print("Correlations with target variable:\n", correlations)

    return correlations

def check_feature_correlations(df):
    """Checks feature correlations."""
    print("Checking feature correlations...")
    encoded_df = df.copy()
    for col in encoded_df.select_dtypes(include=['object']).columns:
        encoded_df[col] = encoded_df[col].astype('category').cat.codes

    correlations = encoded_df.corr()
    print("Feature correlation matrix:\n", correlations)


if __name__ == "__main__":
    preprocess_data()
