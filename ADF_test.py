"""
A simple class that computes and checks the stationarity of a given series
using the Augmented Dickey-Fuller Test
"""
# Import important dependencies 
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# define the class
class Stationarity_test():
    def __init__(self, significance=0.5):
        self.SignificanceLevel = significance
        self.PValue = None
        self.is_stationary = None
    def ADF_test(self, timeseries, printResults= True):
        # adf fuller test
        adfTest = adfuller(timeseries, autolag = 'AIC')

        self.PValue = adfTest[1]

        self.is_stationary = (self.PValue<self.SignificanceLevel)

        if printResults:
            dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
            #Add Critical Values
            for key,value in adfTest[4].items():
                dfResults['Critical Value (%s)'%key] = value
            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)
