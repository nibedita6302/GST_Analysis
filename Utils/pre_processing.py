import pandas as pd

def data_fillna(df_merged):
   
    # Fill NaN values in Column6, Column8 & Column15 with their median
    df_merged['Column6'] = df_merged['Column6'].fillna(df_merged['Column6'].median()) 
    df_merged['Column8'] = df_merged['Column8'].fillna(df_merged['Column8'].median())
    df_merged['Column15'] = df_merged['Column15'].fillna(df_merged['Column15'].median())

    # Fill NaN values in Column3 and Column4 with their mean
    df_merged['Column3'] = df_merged['Column3'].fillna(df_merged['Column3'].mean())
    df_merged['Column4'] = df_merged['Column4'].fillna(df_merged['Column4'].mean())
    
    # For Target 0: Replace NaN in Column5 with mean of Column5
    df_merged.loc[df_merged['target'] == 0, 'Column5'] = df_merged.loc[df_merged['target'] == 0, 'Column5'].\
                                                            fillna(df_merged.loc[df_merged['target'] == 0, 'Column5'].mean())
    # For Target 1: Replace NaN in Column5 with median of Column5
    df_merged.loc[df_merged['target'] == 1, 'Column5'] = df_merged.loc[df_merged['target'] == 1, 'Column5'].\
                                                            fillna(df_merged.loc[df_merged['target'] == 1, 'Column5'].median())
    
    # Handle missing values in Column14
    pure_mean = df_merged[df_merged['Column14'] > -200]['Column14'].mean()
    df_merged['Column14'] = df_merged['Column14'].fillna(df_merged['Column14'].mean())
    
    # Fill NaN values in Column0 with its median
    df_merged['Column0'] = df_merged['Column0'].fillna(df_merged['Column0'].median())
        
    df_merged = df_merged.drop(['Column9'],axis=1) # Drop Column 9
    
    # Return the pre-processed data
    return df_merged


def remove_outliers(df_merged):    
    # Remove rows with 4 or more NaN values
    df_cleaned = df_merged.dropna(thresh=df_merged.shape[1] - 4 + 1)
    
    # Cap values in Column5 at 100
    max_valid_value = df_cleaned[df_cleaned['Column5'] < 100]['Column5'].max()
    df_cleaned.loc[df_cleaned['Column5'] > 100, 'Column5'] = max_valid_value
    
    # Handle extreme values in Column14
    pure_mean = df_cleaned[df_cleaned['Column14'] > -200]['Column14'].mean()
    df_cleaned.loc[df_cleaned['Column14'] < -200, 'Column14'] = pure_mean
    
    # Return clean data
    return df_cleaned

def testing(df):
	print('Hello')
	return df.shape