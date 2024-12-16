import os
import pandas as pd
import click
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

@click.command()
@click.option('--input_path', type=str, required=True, help="Path to the input cleaned CSV file.")
@click.option('--model_output_path', type=str, required=True, help="Path to save the trained model (pickle).")
@click.option('--confusion_matrix_output', type=str, required=True, help="Path to save the confusion matrix plot.")
def train_model(input_path, model_output_path, confusion_matrix_output):
    """
    Reads the cleaned data from the input CSV, trains a Decision Tree model using GridSearchCV, 
    saves the trained model and confusion matrix plot.
    """
    np.random.seed(42)  
    
    try:
       
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"The input file does not exist at: {input_path}")
        
        
        print(f"Reading cleaned data from: {input_path}")
        bank_data = pd.read_csv(input_path)
        
        unknown_columns = bank_data.columns[bank_data.isin(['unknown']).any()]
        cleaned_data = bank_data.apply(lambda col: col.replace('unknown', 'other') if col.dtypes == 'object' else col)

        # Define feature subsets
        numeric_feats = ['age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign']
        categorical_feats = ['job', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
        drop_feats = ['duration', 'month', 'day_of_week', 'pdays', 'marital', 'previous']

       
        X = cleaned_data.drop(columns=drop_feats + ['y']) 
        y = cleaned_data['y'] 

       
        ct = make_column_transformer(
            (StandardScaler(), numeric_feats),  
            (OneHotEncoder(drop="if_binary", sparse_output=False), categorical_feats) features
        )

        
        pipeline = Pipeline(steps=[
            ('preprocessor', ct),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        
        param_grid = {
            'classifier__max_depth': [3, 5, 7, 10, None], 
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 5],  
            'classifier__criterion': ['gini', 'entropy']  
        }

       
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training data size: {X_train.shape}, Test data size: {X_test.shape}")

     
        grid_search.fit(X_train, y_train)

      
        best_model = grid_search.best_estimator_

       
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

     
        print("Best Parameters from Grid Search:", grid_search.best_params_)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", conf_matrix)

        precision = precision_score(y_test, y_pred, pos_label='yes')  
        recall = recall_score(y_test, y_pred, pos_label='yes')
        f1 = f1_score(y_test, y_pred, pos_label='yes')

        
        print("Decision Tree Evaluation with Optimized Hyperparameters:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

       
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


        model_dir = os.path.dirname(model_output_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        with open(model_output_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Model saved to: {model_output_path}")

    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    train_model()
