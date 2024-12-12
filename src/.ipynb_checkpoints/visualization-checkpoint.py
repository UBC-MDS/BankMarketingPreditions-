import altair as alt
import pandas as pd

def generate_scatter_plot(df, x_col, y_col, output_path):
    """
    Generates a scatter plot.

    Args:
        df (pd.DataFrame): The data frame containing the data.
        x_col (str): Column for the x-axis.
        y_col (str): Column for the y-axis.
        output_path (str): Path to save the scatter plot.

    Returns:
        None: Saves the plot as an image.
    """
    chart = alt.Chart(df).mark_point().encode(
        x=alt.X(x_col, title=x_col),
        y=alt.Y(y_col, title=y_col),
        color=alt.Color('y:N', scale=alt.Scale(scheme='category10'))
    ).properties(title=f"Scatterplot of {x_col} vs {y_col}")

    chart.save(output_path)
