import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from sklearn.model_selection import KFold

def train_logistic_model(X, y, model_output_path, cv=5):
    """
    Trains a logistic regression model and saves the model.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        model_output_path (str): Path to save the trained model.
        cv (int): Number of cross-validation folds. Default is 5.

    Returns:
        Pipeline: Trained model pipeline.
    """
    numeric_feats = X.select_dtypes(include=["number"]).columns
    categorical_feats = X.select_dtypes(include=["object"]).columns

    ct = make_column_transformer(
        (StandardScaler(), numeric_feats),
        (OneHotEncoder(drop="if_binary", sparse_output=False), categorical_feats)
    )

    pipeline = Pipeline([
        ('preprocessor', ct),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Use KFold instead of StratifiedKFold
    kfold = KFold(n_splits=cv)
    grid = GridSearchCV(pipeline, param_grid={'classifier__C': [0.01, 0.1, 1]}, cv=kfold)
    grid.fit(X, y)

    with open(model_output_path, 'wb') as f:
        pickle.dump(grid.best_estimator_, f)

    return grid.best_estimator_

if __name__ == "__main__":
