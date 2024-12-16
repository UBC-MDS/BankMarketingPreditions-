import os
import pandas as pd
import click
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

def load_and_clean_data(input_path):
    """
    Reads the cleaned data from the input CSV and cleans it.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The input file does not exist at: {input_path}")
    
    print(f"Reading cleaned data from: {input_path}")
    bank_data = pd.read_csv(input_path)
    
    unknown_columns = bank_data.columns[bank_data.isin(['unknown']).any()]
    cleaned_data = bank_data[~bank_data.isin(['unknown']).any(axis=1)]
    
    return cleaned_data

def prepare_features_and_target(cleaned_data):
    """
    Prepares features and target variable from the cleaned dataset.
    """
    drop_feats = ['duration', 'month', 'day_of_week', 'pdays', 'marital', 'previous']
    X = cleaned_data.drop(columns=drop_feats + ['y'])  # Features excluding target 'y' and drop_feats
    y = cleaned_data['y']  # Target variable
    
    return X, y

def create_column_transformer():
    """
    Creates a column transformer for preprocessing numeric and categorical features.
    """
    numeric_feats = ['age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign']
    categorical_feats = ['job', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    
    ct = make_column_transformer(
        (StandardScaler(), numeric_feats),  # Standard scaling for numeric features
        (OneHotEncoder(drop="if_binary", sparse_output=False), categorical_feats)  # One-hot encoding for categorical features
    )
    
    return ct

def create_pipeline(ct):
    """
    Creates a pipeline for logistic regression with preprocessing steps.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', ct),
        ('classifier', LogisticRegression(solver='liblinear'))  # Using 'liblinear' solver for small datasets
    ])
    
    return pipeline

def perform_grid_search(pipeline, X_train, y_train):
    """
    Performs GridSearchCV to find the best hyperparameters.
    """
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'classifier__penalty': ['l1', 'l2'],  # Regularization type
        'classifier__solver': ['liblinear'],  # Solver for logistic regression
        'classifier__max_iter': [100, 200, 300]  # Maximum number of iterations for convergence
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search

def evaluate_model(y_test, y_pred):
    """
    Evaluates the model by calculating accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='yes')
    recall = recall_score(y_test, y_pred, pos_label='yes')
    f1 = f1_score(y_test, y_pred, pos_label='yes')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    return accuracy, precision, recall, f1

def plot_confusion_matrix(conf_matrix, confusion_matrix_output):
    """
    Plots and saves the confusion matrix as a heatmap.
    """
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['No', 'Yes'], columns=['No', 'Yes'])
    conf_matrix_df = conf_matrix_df.reset_index().melt(id_vars="index")
    conf_matrix_df.columns = ['Actual', 'Predicted', 'Count']
    
    chart = alt.Chart(conf_matrix_df).mark_rect().encode(
        x='Predicted:N',
        y='Actual:N',
        color='Count:Q',
        tooltip=['Actual:N', 'Predicted:N', 'Count:Q']
    ).properties(
        title="Confusion Matrix"
    )
    
    chart.save(confusion_matrix_output)

def save_model(pipeline, model_output_path):
    """
    Saves the trained model to the specified path.
    """
    model_dir = os.path.dirname(model_output_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    with open(model_output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved to: {model_output_path}")

@click.command()
@click.option('--input_path', type=str, required=True, help="Path to the input cleaned CSV file.")
@click.option('--model_output_path', type=str, required=True, help="Path to save the trained model (pickle).")
@click.option('--confusion_matrix_output', type=str, required=True, help="Path to save the confusion matrix plot.")
def train_model(input_path, model_output_path, confusion_matrix_output):
    """
    Main function to train the model, evaluate, and save the results.
    """
    try:
        # Step 1: Load and clean data
        cleaned_data = load_and_clean_data(input_path)
        
        # Step 2: Prepare features and target
        X, y = prepare_features_and_target(cleaned_data)
        
        # Step 3: Create column transformer
        ct = create_column_transformer()
        
        # Step 4: Create pipeline
        pipeline = create_pipeline(ct)
        
        # Step 5: Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training data size: {X_train.shape}, Test data size: {X_test.shape}")
        
        # Step 6: Perform grid search
        grid_search = perform_grid_search(pipeline, X_train, y_train)
        print("Best hyperparameters found: ", grid_search.best_params_)
        
        # Step 7: Evaluate model
        y_pred = grid_search.predict(X_test)
        evaluate_model(y_test, y_pred)
        
        # Step 8: Plot confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(conf_matrix, confusion_matrix_output)
        
        # Step 9: Save the model
        save_model(grid_search.best_estimator_, model_output_path)
    
    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    train_model()
