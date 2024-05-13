Data-driven HT-ATES model 
Created by David Geerts for the PUSH-IT project. 
More information about this model can be found in the paper published by David Geerts (to be published).
If the model is used in any work, please cite that paper.

The data-driven model is created to skip the numerical modelling, which is often time consuming.
This model create the temperature profile of an HT-ATES system in a computationally efficient manner to be used in larger system modelling.

To use it download all the stuff in a folder and import the ATES_obj_publish file.
An example of how to use it is within that file. 

The "Predict_REFF_boostedregression.pkl" file is an extreme gradient boosting algorithm (machine learning) that predicts the recovery efficiency based on the input parameters.
The "results_filtered' (parquet file) contains all the data generated using a MODFLOW model. 

This data driven model was tested using the following package versions
Python 3.9.18
Numpy 1.26.4
Pandas 2.2.1
Matplotlib 3.8.3 
Pyarrow 15.0.1
Joblib 1.3.2
Scikit-learn 1.3.0
