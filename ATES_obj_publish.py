import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import time 

class ATES_obj:
    """
    Data-driven ATES model.
    Parameters
    ----------
    supplier : str
        Supplier of the ATES system.
    thickness : float, optional
        Thickness of the aquifer in meters. Default is 20.
    porosity : float, optional
        Porosity of the aquifer. Default is 0.3.
    kh : float, optional
        Hydraulic conductivity of the aquifer in m/s. Default is 10.
    ani : float, optional
        Anisotropy of the aquifer. Default is 10.
    T_ground : float, optional
        Temperature of the ground in degrees Celsius. Default is 10.
    start_full_volume : float, optional
        Number between 0 and 1 indicating the ratio of volume that is injected before the first extraction period (see paper attached). Default is 1.
    timing : bool, optional
        Flag indicating whether to enable timing for performance evaluation. Default is False.

    
    Attributes
    ----------
    data : DataFrame
        Loaded data from the 'results_filtered' parquet file.
    DOE_data : DataFrame
        Processed Design of Experiment data containing unique combinations of input parameters, further defined in a paper to be published
    last_year_data : DataFrame
        Subset of data containing records from the last year of ATES operation, being the 8th year.

    Methods
    -------
    __init__
        Initializes the ATES_obj with specified parameters and loads required data.
    initialize(volume, T_in, len_timestep)
        Initializes the ATES system with specified volume, injected temperature, and time step.
    predict_reff(volume, T_in)
        Predicts the recovery efficiency based on a machine learning model.
    nearest_neighbour(T_in, reff, volume)
        Finds the nearest neighbors in the data for temperature prediction.
    predict_temp_out(T_in, reff, volume)
        Predicts the outlet temperature based on nearest neighbors.
    correct_for_volume(volume, temp_out, len_timestep)
        Corrects the outlet temperature for the specified volume and time step.

    Notes
    -----
    This class encapsulates the functionality of an ATES system, including data handling, temperature prediction, and heat calculation.
    """
    
    def __init__(self, thickness = 20, porosity = 0.3, kh = 10, 
                 ani = 10, T_ground = 10,
                 start_full_volume = 1,timing=False):
        
        # Save data
        self.name = 'ATES'
        self.control = 'storage'
        self.type = 'supply'
        self.thickness = thickness
        self.por = porosity
        self.kh = kh
        self.ani=ani
        self.T_g = T_ground
        self.start_full_volume = start_full_volume
        self.timing=timing
        
        
        #Start the timer for timing 
        if timing:
            start = time.time()
        
        # Get data from earlier research, saved in parquet file and manipulate it 
        self.data = pd.read_parquet('results_filtered')
        self.data.drop(self.data.columns[10], axis=1,inplace=True)
        self.data.drop(self.data.columns[5], axis=1, inplace = True)
        self.data.drop(self.data.columns[3], axis=1, inplace = True)
        self.data.drop(self.data[(self.data["Efficiency_hotwell_lastyear"]>1)].index, inplace=True)
        
        # Find time for looading data
        if timing:
            print('Loading parquet data took {}s'.format(time.time() - start))
        
        # Filter for the last year data
        self.last_year_data = self.data.drop(self.data[(self.data["Day"]<2554)].index)

    def initialize(self,volume,T_in, len_timestep):
        """
        Initializes the ATES_obj instance with the given volume, injected temperature, and length of timestep.
        This should be called when adding the ATES system
        
        Parameters
        ----------
        volume : float
            Volume of the injected water in cubic meters.
        T_in : float
            Injected temperature in degrees Celsius.
        len_timestep : float
            Length of each timestep in seconds.
        
        This method involves the following steps:
        1. Predicts the recovery efficiency (reff) using the predict_reff method.
        2. Predicts the outlet temperature (temp_out) using the predict_temp_out method.
        3. Corrects the data for the given volume and temp_out using the correct_for_volume method.
        
        Parameters
        ----------
        volume : float
            Volume of the aquifer in cubic meters.
        T_in : float
            Injected temperature in degrees Celsius.
        len_timestep : float
            Length of each timestep in seconds.
        """
        #%% Step 1: Predict recovery efficiency using ML
        if self.timing:
            start = time.time()
            
        reff = self.predict_reff(volume, T_in)
        self.Reff= reff

        if self.timing:
            print('Predicting reff took {}s'.format(time.time() - start))
        
        if self.timing:
            start = time.time()
        
        #%% Step 2: Predict outlet temperature based on the Reff, T_in, V
        temp_out = self.predict_temp_out(T_in, reff, volume)
        
        if self.timing:
            print('Nearest neighbour search took {}s'.format(time.time() - start))
        
        
        if self.timing:
            start = time.time()
            
        #%% Step 3: Correct data for volume
        if self.start_full_volume != 0.5:
            temp_out = self.Correct_half_volume(temp_out)
        
        self.correct_for_volume(volume, T_in,temp_out, len_timestep)
        
        if self.timing:
            print('Manipulation took {}s'.format(time.time() - start))
        
    def predict_reff(self, volume, T_in):
        """
        Predicts recovery efficiency based on the given volume and injected temperature.
        
        Parameters
        ----------
        volume : float
            Volume of the injected water in cubic meters.
        T_in : float
            Injected temperature in degrees Celsius.
        
        Returns
        -------
        float
            Predicted recovery efficiency of the aquifer.
        
        Notes
        -----
        - The Ml model is generated based on the data found in self.data. It uses boostedregression from scikit.learn
        - The predicted recovery efficiency is obtained and stored in the instance variable Reff.
        """
        # Load the ML model
        model = joblib.load("Predict_REFF_boostedregression.pkl")        


        # Prepare inputs for prediction
        Reff = model.predict(pd.DataFrame({'Porosity':self.por,
                                           "Volume" :volume,
                                           "T_injected" :T_in,
                                           "T_ground":self.T_g,
                                           "thickness aquifer" :self.thickness,
                                           'Hydraulic conductivity aquifer':self.kh,
                                           'anisotropy':self.ani},index=[0]))
        # Get a float out, not a list 
        Reff=Reff[0]
        
        if Reff < 0.4:
            print("Low recovery (<0.4) efficiency consider improving parameters")

        #
        if volume<50000:
            print("Volume injected into aquifer very low, consider increasing")

        return Reff
    
    def nearest_neighbour(self,T_in,reff,volume):
        """
        Finds the nearest neighbors in the dataset based on input parameters.
    
        Parameters
        ----------
        T_in : float
            Injected temperature in degrees Celsius.
        reff : float
            Recovery efficiency.
        volume : float
            Volume of the aquifer in cubic meters.
    
        Returns
        -------
        Tuple
            A tuple containing the indices of the nearest neighbors and their total distances.
    
        Notes
        -----
        - Calculates relative distances for temperature, recovery efficiency, ground temperature, and volume.
        - Computes the total distance as the Euclidean norm of the relative distances.
        - Finds the indices of the nearest neighbors and their total distances.
        """
        #Calculate relative distances
        relative_distance_1 = abs(((self.data["T_injected_hot"])-(T_in))/90)#(T_in))
        relative_distance_2 = abs((self.data["Efficiency_hotwell_lastyear"]-reff)/0.9)#/reff)
        relative_distance_3 = abs((self.data["T_ground"]-self.T_g)/30)#self.T_g)
        # Legacy aspect
        #relative_distance_4 = 0#abs((self.data["Volume"]-volume)/volume)
        #relative_distance_5 = 0#((self.data["anisotropy"]-self.ani)/self.ani)**2+((self.data['Porosity']-self.por)/self.por)**2+((self.data["thickness aquifer"]-self.thickness)/self.thickness)**2+((self.data['Hydraulic conductivity aquifer']-self.kh)/self.kh)**2


        # Compute the total distance
        total_distance = np.sqrt(relative_distance_1+relative_distance_2+relative_distance_3)

        # Find the indices of the nearest neighbors
        lowest = total_distance[total_distance==total_distance.min()].index

        return lowest,total_distance
    
    def predict_temp_out(self,T_in,reff,volume):
        """
        Predicts the outlet temperature based on the nearest neighbors.
        
        Parameters
        ----------
        T_in : float
            Injected temperature in degrees Celsius.
        reff : float
            Recovery efficiency.
        volume : float
            Volume of the aquifer in cubic meters.
        
        Returns
        -------
        pandas.Series
            Predicted outlet temperature for all of the 8 years.
        
        Notes
        -----
        - Finds the nearest neighbors and their total distances using the nearest_neighbour method.
        - Retrieves the outlet temperature of the nearest neighbors.
        - Normalizes the temperature values.
        - Stores the predicted outlet temperature in the instance variable temp_out.
        - Returns the predicted outlet temperature.
        """
        # Find the nearest neightbours and their total distance
        lowest,total_distance = self.nearest_neighbour(T_in,reff,volume)
        
        # Retrieve the outlet temperature of the nearest neighbors
        temp_out = self.data.loc[lowest]["Outlet_T_hotwell"]

        # Correct the temperature if nearest neighbour temperature is not exactly the same
        temp_out = temp_out/(temp_out.iloc[-1]/T_in)

        #Store and return the outlet temperature
        self.temp_out=temp_out
        return temp_out

    def correct_for_volume(self,volume, T_in,temp_out,len_timestep):
        """
        Corrects the temperature output based on the provided volume.
        This is based on the data calculation, which used periods of 1 week.
        
        Parameters
        ----------
        volume : float
            Volume in cubic meters per year.
        temp_out : pd.Series
            Temperature output.
        len_timestep : int
            Length of each time step in seconds.
        
        Notes
        -----
        - Computes flow based on provided volume.
        - Adjusts the temperature output accordingly.
        """
        #Volume in m^3 per year
        perlen=7
        PerPerYear= int(round(365/perlen, 0))

        # Calculate flow
        flow = self.calculate_flow(volume, PerPerYear)  
        
        # Set up time index
        index = pd.Series(np.linspace(0,416*24*7,417),dtype = int)


        #Calculate flow
        flow = np.cumsum(flow.clip(min=0))
        
        # Calculate temperature out based on the volume
        temp_out.reset_index(drop=True,inplace=True)
        temp_out = pd.concat([temp_out,pd.Series(flow,name="flow")],axis=1)
        temp_out = temp_out.set_index(index)

        # Interpolate between missing values
        temp_out = temp_out.reindex(range(int(temp_out.index.min()),int(temp_out.index.max()))).interpolate()
        
        # Interpolation and taking the nearest neighbour messes with the Reff
        # Therefore correct for the Reff again. Reff of ML is quite accurate, so stick to it.
        after_inter = sum((temp_out.iloc[-(52*24*7):,0]-self.T_g)*np.diff(temp_out.iloc[-(52*24*7):,1],prepend=min(temp_out.iloc[-(52*24*7):,1])))/((T_in-self.T_g)*volume)
        factor = 5
        factor_save=5
        while factor < 0.99 or factor >1.01 :
            factor = self.Reff/after_inter
            if abs(1/factor_save-1)<abs(factor-1):
                factor = ((factor)-1)*0.5+1
            temp_out.loc[:,"Outlet_T_hotwell"]=(temp_out.loc[:,"Outlet_T_hotwell"]-self.T_g)*(((factor-1)*1)+1)+self.T_g
            after_inter = sum((temp_out.iloc[-(52*24*7):,0]-self.T_g)*np.diff(temp_out.iloc[-(52*24*7):,1],prepend=min(temp_out.iloc[-(52*24*7):,1])))/((T_in-self.T_g)*volume)
            factor_save=factor
        
        #Save it
        self.output_t = temp_out.copy()
        self.output_t_firstyear = self.output_t.head(8760).copy()
        self.output_t_lastyear = self.output_t.tail(8760).copy()
                
        # Correct flow of the last year, for earlier flows
        self.output_t_lastyear.loc[:,"flow"] = self.output_t_lastyear["flow"] - min(self.output_t_lastyear["flow"]) 
        difference = np.diff(self.output_t_lastyear["flow"],prepend= 0)
        T_ave = sum((self.output_t_lastyear.loc[:,"Outlet_T_hotwell"])*difference)/sum(difference)
        self.output_t_lastyear = self.output_t_lastyear[difference>0]
        Reff_calc= (T_ave-self.T_g)/(T_in-self.T_g)

    def calculate_flow(self, volume, PerPerYear):
        """
        Calculates the flow based on the provided volume.

        Parameters
        ----------
        volume : float
            Volume in cubic meters per year.
        weeks_per_year : int
            Number of weeks in a year.

        Returns
        -------
        np.ndarray
            Array representing the calculated flow.
        """
        sum_sine = 0
        periods_per_half_year = int(PerPerYear / 2)
        flow = np.zeros(417)

        # Calculate sum of sine values
        for i in range(periods_per_half_year):
            sine = np.sin(np.pi * i / periods_per_half_year)
            sum_sine += sine

        # Calculate flow
        for j in range(len(flow)):
            flow[j] = round(np.cos(np.pi * j / periods_per_half_year) / sum_sine * (-1) * volume, 0)

        return flow
    
    def func_fit(self,x, a, b,c): # polytrope equation from zunzun.com
        return a+b/(2**(x/c))
    
    def Correct_half_volume(self, temp_out):
        """
        Corrects the temperature output for half-volume cycles.
        
        Parameters
        ----------
        temp_out : pandas.Series
            Temperature output data.
      
        Returns
        -------
        pandas.Series
            Corrected temperature output data.
     
        Notes
        -----
        The function identifies half-volume cycles in the temperature output data and corrects them by fitting a curve
        and adjusting the values accordingly.      
        """
        
        # Identify half-volume cycles
        index =np.reshape(np.array(argrelextrema(np.array(temp_out),np.less,order=20)),8)
        output = np.array(temp_out.iloc[index])
        index = ((index-13)/52)
        
        # Fit a curve to the identified cycles
        popt, pcov = curve_fit(self.func_fit, index, output,p0=[35,-8.2,0.77])
        
        # Correct half-volume cycles for the first five years, after which the data is more accurate
        for j in range(5):
            min_half_cycle = temp_out-max(temp_out)
            min_full_cycle = self.func_fit(j+self.start_full_volume,*popt)-max(temp_out)
            min_value_half_cycle = output[j]-max(temp_out)
            factor = min_full_cycle/min_value_half_cycle
            temp_out[j*52:(j+1)*52] = min_half_cycle[j*52:(j+1)*52]*factor+max(temp_out)
        self.temp_out=temp_out
        return temp_out
        
# if __name__ == "__main__":
#     # Parameters
#     thickness_aquifer = 20 #[m] Thickness of the aquifer (assuming homogenous and constant thickness)
#     porosity = 0.2 #[-] porosity aquifer
#     horizontal_conductivity = 10  #[m day^-1] Horizontal hydraulic conductivity
#     anisotropy =10 #[-] Horizontal hydraulic conductivity/vertical hydraulic conductivity
#     ground_temperature = 15 #[degrees C] Undisturbed ground temperature
#     supplier = 0 
#     ATES = ATES_obj(thickness=thickness_aquifer, porosity=porosity,kh=horizontal_conductivity, 
#                     ani=anisotropy,T_ground=ground_temperature,start_full_volume=1)
    
#     Volume = 100000 #m^3/year, volume injected as well as extracted (assuming mass balance needs to be preserved)
#     Temp_in = 70 #[degrees C] Temperature of the water going in the aquifer
#     ATES.initialize(Volume, Temp_in, 3600) #Generates values for T_out
    
#     plt.plot(ATES.output_t.loc[:,"flow"],ATES.output_t.loc[:,"Outlet_T_hotwell"])
#     plt.xlabel("Volume (m^3) extracted")
#     plt.ylabel("Temperature out of ATES")
#     print("Predicted recovery efficiency = ",ATES.Reff) 
