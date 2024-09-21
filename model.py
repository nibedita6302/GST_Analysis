import numpy as np
import pandas as pd
import joblib 
from sklearn.linear_model import LogisticRegression
import Utils.pre_processing as pp
  
def train_model(df):
    X = df.iloc[:,1:-1].values
    Y = df.iloc[:,-1].values
    # LR model with best hyperparameters
    model = LogisticRegression(C=1, max_iter=200, penalty='l2', solver='lbfgs') # parameterized LR model
    model.fit(X, Y) # Fit cleaned dataset to model for training
    
    # Save the model to a file
    model_filename = "new_saved_model.joblib"
    joblib.dump(model, model_filename)

    print(f"Trained Model saved to {model_filename}")
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

# Create and train model with best parameter found during hyperparameter tuning 
train_model(df_cleaned) # using cleaned data


                         