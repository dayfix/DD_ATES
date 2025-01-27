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
    Data_clean = Data.iloc[:,0:9].drop_duplicates()
    #Data_analysis = manipulate_Data(Data_clean)


    Data_clean.drop(["groundwater  flow"],axis=1,inplace=True)
    Data_clean.reset_index(drop=True,inplace=True)
    return Data_clean,Data

Data_analysis,raw_data = Data_unpacking(file_name='results_AXI')
y = Data_analysis["Efficiency_well_lastyear"]
X = Data_analysis[['Porosity', 'Volume', 'T_injected_hot', 'T_ground', 'thickness aquifer',
       'Hydraulic conductivity aquifer', 'anisotropy']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)#,random_state=1)


import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

reg=xgb_model
reg.fit(X_train, y_train)
Predictions = reg.predict(X_test)
Predictions = Predictions.reshape(len(Predictions))
RMSE = np.sqrt(sum((Predictions - y_test)**2)/len(Predictions))
print("RMSE = "+str(RMSE))#
#joblib.dump(reg,"Predict_REFF_boostedregression.pkl")



