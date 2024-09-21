import pandas as pd
import joblib 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import Utils.pre_processing as pp

def hyper_parameter_tuning(df):
    X = df.iloc[:,1:-1].values
    Y = df.iloc[:,-1].values
    # Define the parameter grid
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'], # Type of regularization
        'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'solver': ['lbfgs', 'saga'],  # Solver algorithms
        'max_iter': [100, 200]  # Maximum number of iterations
    }

    # Initialize the model
    model = LogisticRegression()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    
    # Fit GridSearchCV
    grid_search.fit(X, Y)
    print('Please wait, this might take several minutes...')
    
    # Get the best parameters
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"Best Parameters: {best_params}")
    
    # Save the model to a file
    model_filename = "saved_best_model.joblib"
    joblib.dump(best_model, model_filename)

    print(f"Hyperparameter Tuning completed! Best Model saved to {model_filename}")
    return 

def clean_data(x_train_path, y_train_path):
    # Read X and Y data from path
    X = pd.read_csv(x_train_path)
    Y = pd.read_csv(y_train_path)
    # Merge X and Y to one dataframe
    df_merged = pd.merge(X, Y[['ID', 'target']], on='ID')
    # Handle NaN or missing values
    df_merged = pp.data_fillna(df_merged)
    # Handle Outliers
    df_cleaned = pp.remove_outliers(df_merged)
    
    # Return clean dataframe
    return df_cleaned

# __main__
X_train_path = input('Enter path for X Training Data (CSV):')
Y_train_path = input('Enter path for Y Training Data (CSV):')

# Clean Training Data
df_cleaned = clean_data(X_train_path, Y_train_path)

# Create and train model with cleaned data
hyper_parameter_tuning(df_cleaned) 