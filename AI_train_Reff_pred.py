# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:53:05 2023

@author: 6100430
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# importing the libraries

import joblib
import matplotlib.pyplot as plt



def Data_unpacking(file_name = 'results_filtered'):        
    """
    Unpacks and cleans the data for analysis.
    
    Args:
       file_name (str): Name of the parquet file to read. Default is 'results_filtered'.

    Returns:
        pandas.DataFrame: Cleaned and filtered data for analysis.
    
    """
    Data = pd.read_parquet(file_name)
    Data_clean = Data.iloc[:,0:11].drop_duplicates()
    Data_analysis = manipulate_Data(Data_clean)
    Data_analysis.drop(Data_analysis[(Data_analysis["Efficiency_well_lastyear"]>0.86)].index, inplace=True)
    Data_analysis.reset_index(drop=True,inplace=True)
    return Data_analysis

def manipulate_Data(Data):
    """
    Manipulates the given Data by dropping columns and renaming columns.

    Args:
        Data (pandas.DataFrame): The input data to be manipulated.

    Returns:
        pandas.DataFrame: The manipulated data.

    """
    # Drop columns and assign to data_man_1
    data_man_1 = Data.drop(Data.columns[9], axis=1)
    data_man_1 = data_man_1.drop(Data.columns[2], axis=1)

    # Rename columns in data_man_1
    data_man_1.rename(columns={'T_injected_cold': 'T_injected', 'Efficiency_coldwell_lastyear': 'Efficiency_well_lastyear'}, inplace=True)

    # Drop columns and assign to data_man_2
    data_man_2 = Data.drop(Data.columns[10], axis=1)
    data_man_2 = data_man_2.drop(Data.columns[3], axis=1)

    # Rename columns in data_man_2
    data_man_2.rename(columns={'T_injected_hot': 'T_injected', 'Efficiency_hotwell_lastyear': 'Efficiency_well_lastyear'}, inplace=True)

    # Drop a column in data_man_2 and assign to Data_clean
    Data_clean = data_man_2.drop(data_man_2.columns[4], axis=1)

    return Data_clean


Data_analysis = Data_unpacking(file_name='results_filtered')
y = Data_analysis["Efficiency_well_lastyear"]
X = Data_analysis[['Porosity', 'Volume', 'T_injected', 'T_ground', 'thickness aquifer',
       'Hydraulic conductivity aquifer', 'anisotropy']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)#,random_state=1)


import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

reg = GradientBoostingRegressor(n_estimators=3000)
reg=xgb_model
reg.fit(X_train, y_train)
Predictions = reg.predict(X_test)
Predictions = Predictions.reshape(len(Predictions))
RMSE = np.sqrt(sum((Predictions - y_test)**2)/len(Predictions))
print("RMSE = "+str(RMSE))
joblib.dump(reg,"Predict_REFF_boostedregression.pkl")



