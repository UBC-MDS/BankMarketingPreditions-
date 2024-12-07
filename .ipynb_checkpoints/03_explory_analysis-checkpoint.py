import os
import pandas as pd
import altair as alt
import click

@click.command()
@click.option('--cleaned_data_path', type=str, required=True, help="Path to the cleaned data file.")
@click.option('--output_prefix', type=str, required=True, help="Prefix for saving the output visualization files.")
def generate_eda(cleaned_data_path, output_prefix):
    """
    Generates exploratory data visualizations using the cleaned data and saves them to files.
    """
    try:
        # Load the cleaned dataset
        if not os.path.isfile(cleaned_data_path):
            raise FileNotFoundError(f"The cleaned data file does not exist at: {cleaned_data_path}")
        print(f"Reading cleaned data from: {cleaned_data_path}")
        bank_data = pd.read_csv(cleaned_data_path)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Univariate distributions for numeric variables
        print("Generating univariate distributions for numeric variables...")
        for col in ['age', 'duration', 'campaign', 'previous']:
            chart = alt.Chart(bank_data).mark_bar().encode(
                alt.X(col, bin=alt.Bin(maxbins=30)),
                alt.Y('count()'),
                alt.ColorValue('steelblue')
            ).properties(title=f'Distribution of {col}')
            chart.save(f"{output_prefix}_{col}_dist.png")

        # Univariate distribution for categorical variables
        print("Generating univariate distributions for categorical variables...")
        target_chart = alt.Chart(bank_data).mark_bar().encode(
            x=alt.X('y:N', title='Target Variable (y)'),
            y=alt.Y('count()', title='Count'),
            color=alt.Color('y:N', scale=alt.Scale(scheme='category10'))
        ).properties(title='Target Variable Distribution')
        target_chart.save(f"{output_prefix}_target_dist.png")

        # Pairwise correlations for quantitative variables
        print("Generating pairwise correlations for quantitative variables...")
        corr = bank_data.select_dtypes(include=['number']).corr()
        corr_chart = alt.Chart(corr.reset_index().melt('index')).mark_rect().encode(
            x=alt.X('index:N', title='Feature'),
            y=alt.Y('variable:N', title='Feature'),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='blueorange')),
            tooltip=['index', 'variable', 'value']
        ).properties(title='Correlation Heatmap')
        corr_chart.save(f"{output_prefix}_correlation.png")

        # Pairwise scatterplots for high-correlation variables
        print("Generating pairwise scatterplots for high-correlation variables...")
        high_corr_columns = ["age", "duration", "campaign", "previous", "y"]
        scatter_data = bank_data[high_corr_columns].sample(n=300, random_state=42)
        scatter_chart = alt.Chart(scatter_data).mark_point().encode(
            x=alt.X('age', title='Age'),
            y=alt.Y('duration', title='Duration'),
            color=alt.Color('y:N', scale=alt.Scale(scheme='category10'))
        ).properties(title='Scatterplot of Age vs Duration')
        scatter_chart.save(f"{output_prefix}_scatterplot.png")

        print("EDA visualizations generated and saved successfully!")

    except Exception as e:
        print(f"Error during EDA visualization generation: {e}")


if __name__ == "__main__":
    generate_eda()
