import pytest
import pandas as pd
from src.visualization import generate_scatter_plot
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


def test_generate_scatter_plot(tmp_path):
    data = pd.DataFrame({
        'age': [25, 30],
        'duration': [100, 200],
        'y': ["yes", "no"]
    })

    output_path = tmp_path / "scatter.png"
    generate_scatter_plot(data, 'age', 'duration', output_path)

    assert output_path.exists()
