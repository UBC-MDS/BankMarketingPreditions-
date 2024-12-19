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


