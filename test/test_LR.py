import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock, mock_open
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import pickle
import shutil
from model_LR import ( evaluate_model,  save_model)



def test_evaluate_model():
    """Test the evaluate_model function."""
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]

    with patch('builtins.print') as mock_print:
        accuracy, precision, recall, f1 = evaluate_model(y_true, y_pred)
        
        assert accuracy == 0.8
        assert precision == 1.0
        assert recall == 0.6666666666666666
        assert f1 == 0.8
        
        mock_print.assert_any_call(f"Accuracy: 0.80")
        mock_print.assert_any_call(f"Precision: 1.00")
        mock_print.assert_any_call(f"Recall: 0.67")
        mock_print.assert_any_call(f"F1 Score: 0.80")



def test_save_model():
    """Test the save_model function."""
    # Create a dummy model
    model = LogisticRegression(solver='liblinear')
    model.fit([[1, 2, 3]], [1]) 
    
    model_output_path = 'model.pkl'
    
    with patch('builtins.open', mock_open()) as mock_file:
        save_model(model, model_output_path)
        mock_file.assert_called_once_with(model_output_path, 'wb')
        assert os.path.exists(model_output_path) 




    os.remove(mock_csv_file)
    os.remove(model_output_path)
    os.remove(confusion_matrix_output)


