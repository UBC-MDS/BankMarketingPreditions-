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

@click.command()
@click.option('--input_path', type=str, required=True, help="Path to the input cleaned CSV file.")
@click.option('--model_output_path', type=str, required=True, help="Path to save the trained model (pickle).")
@click.option('--confusion_matrix_output', type=str, required=True, help="Path to save the confusion matrix plot.")
def train_model(input_path, model_output_path, confusion_matrix_output):
    """
    Reads the cleaned data from the input CSV, trains a Logistic Regression model using GridSearchCV, 
    saves the trained model and confusion matrix plot.
    """
    try:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"The input file does not exist at: {input_path}")
        
 
        print(f"Reading cleaned data from: {input_path}")
        bank_data = pd.read_csv(input_path)
        

        unknown_columns = bank_data.columns[bank_data.isin(['unknown']).any()]
        cleaned_data = bank_data[~bank_data.isin(['unknown']).any(axis=1)]
        

        drop_feats = ['duration', 'month', 'day_of_week', 'pdays', 'marital', 'previous']
        X = cleaned_data.drop(columns=drop_feats + ['y'])  # Features excluding target 'y' and drop_feats
        y = cleaned_data['y']  # Target variable


        numeric_feats = ['age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign']
        categorical_feats = ['job', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
        

        ct = make_column_transformer(
            (StandardScaler(), numeric_feats),  # Standard scaling for numeric features
            (OneHotEncoder(drop="if_binary", sparse_output=False), categorical_feats)  # One-hot encoding for categorical features
        )


        pipeline = Pipeline(steps=[
            ('preprocessor', ct),
            ('classifier', LogisticRegression(solver='liblinear'))  # Using 'liblinear' solver for small datasets
        ])
        

        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
            'classifier__penalty': ['l1', 'l2'],  # Regularization type
            'classifier__solver': ['liblinear'],  # Solver for logistic regression
            'classifier__max_iter': [100, 200, 300]  # Maximum number of iterations for convergence
        }


        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training data size: {X_train.shape}, Test data size: {X_test.shape}")


        grid_search.fit(X_train, y_train)


        print("Best hyperparameters found: ", grid_search.best_params_)

 
        y_pred = grid_search.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)


        print("Logistic Regression Evaluation:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:\n", conf_matrix)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='yes')
        recall = recall_score(y_test, y_pred, pos_label='yes')
        f1 = f1_score(y_test, y_pred, pos_label='yes')

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

       # Create confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Create a DataFrame for confusion matrix (for Altair)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=['No', 'Yes'], columns=['No', 'Yes'])
        conf_matrix_df = conf_matrix_df.reset_index().melt(id_vars="index")
        conf_matrix_df.columns = ['Actual', 'Predicted', 'Count']

        # Create the Altair chart (heatmap-like)
        chart = alt.Chart(conf_matrix_df).mark_rect().encode(
            x='Predicted:N',
            y='Actual:N',
            color='Count:Q',
            tooltip=['Actual:N', 'Predicted:N', 'Count:Q']
        ).properties(
            title="Confusion Matrix"
        )

        # Save the Altair plot as a PNG or HTML file
        chart.save(confusion_matrix_output)

        # Save the model
        model_dir = os.path.dirname(model_output_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        with open(model_output_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Model saved to: {model_output_path}")

    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    train_model()
