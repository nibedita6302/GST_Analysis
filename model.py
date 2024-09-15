
import pandas as pd
import numpy as np
import joblib
from Utils import pre_processing as pp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

#C:\Users\Ashis Sardar\Desktop\GST Hack\Test_20\Test_20\Test_20\X_Test_Data_Input.csv
#C:\Users\Ashis Sardar\Desktop\GST Hack\Test_20\Test_20\Test_20\Y_Test_Data_Target.csv

def model_test(X_test_path, Y_test_path):
    # Load test data
    X_test = pd.read_csv(X_test_path)
    Y_test = pd.read_csv(Y_test_path)

    df_merged = pd.merge(X_test, Y_test, on='ID') # Merge X and Y on Column ID

    df_cleaned_test = pp.data_fillna(df_merged) # pre-processing and return the test data

    # Seperate X and Y from dataframe
    X_test_cleaned = df_cleaned_test.iloc[:,1:-1].values # All columns except ID 
    Y_test_cleaned = df_cleaned_test.iloc[:,-1].values # Only Target Column
    
    # scaler = joblib.load('StandardScaler_v1.joblib')
    # X_test_cleaned = scaler.transform(X_test_cleaned) # Standardization of X
    

    # Load saved Logistic Regression Model
    model = joblib.load("logistic_regression_parameterized_v1o2.joblib")
    
    # Predict on the test set
    y_pred = model.predict(X_test_cleaned) # Prediction (0 or 1)
    y_proba = model.predict_proba(X_test_cleaned) # Prediction Probability -> [0,1]

    # Evaluate 
    accuracy = accuracy_score(Y_test_cleaned, y_pred) # Accuracy
    report = classification_report(Y_test_cleaned, y_pred) # Classification Report
    cm = confusion_matrix(Y_test_cleaned, y_pred) # Confusion Matrix
    logloss = log_loss(Y_test_cleaned, y_proba) # Log Loss

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Log Loss: {logloss}")
    
    choice = int(input('''Do you want to save the predictions?
                   0 -> No / 1-> Yes: '''))
    if choice==1:
        fileName = input("Input File Name:")
        # Get ID
        ID = df_cleaned_test.iloc[:,0:1].values
        # Stack them horizontally to create a (10, 3) array
        combined_arr = np.column_stack((ID,Y_test_cleaned, y_pred, y_proba))

        # Save the combined array as a CSV file
        np.savetxt(fileName, combined_arr, delimiter=',', fmt='%s,%d,%d,%.4f,%.4f', \
                    header='ID,Y_True,Y_Prediction,0_Probability,1_Probability', comments='')
        return 'Executed & Saved'
    return 'Only Executed'

# __main__
X_test_path = input('Enter path for X Test Data (CSV):')
Y_test_path = input('Enter path for Y Test Data (CSV):')
comment = model_test(X_test_path,Y_test_path)
print(comment)

    