import pytest
import pandas as pd
from src.modeling import train_logistic_model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_train_logistic_model(tmp_path):
    data = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'job': ["admin.", "technician", "manager", "admin."],
        'y': [1, 0, 1, 0]
    })
    X = data[['age', 'job']]
    y = data['y']

    model_path = tmp_path / "model.pkl"
    model = train_logistic_model(X, y, model_path, cv=2)  # 2 splits are now possible

    assert model is not None
    assert model_path.exists()

