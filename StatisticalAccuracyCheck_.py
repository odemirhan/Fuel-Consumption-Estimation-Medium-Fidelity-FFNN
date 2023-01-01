# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:15:20 2022

@author: Engineering
"""

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from glob import glob

bestmodels=glob("Model Results/1/*.csv")
#modelname="ParamB_FP_AirDistanceFP_64_0.001_200_4layers_MSE"
for cntmodel in range(len(bestmodels)):
    print(bestmodels[cntmodel])
    DF=pd.read_csv(bestmodels[cntmodel])

    dfnumpy=DF.to_numpy()
    yhat=DF[["YHAT"]].to_numpy()
    y=DF[["Y"]].to_numpy()
    
    mae = metrics.mean_absolute_error(y, yhat)
    mape = metrics.mean_absolute_percentage_error(y, yhat)
    mse = metrics.mean_squared_error(y, yhat)
    rmse = np.sqrt(mse) # or mse**(0.5)  
    r2 = metrics.r2_score(y,yhat)
    x=(y-yhat)/y
    threesigma_boundaries=[x.mean()-3*x.std() ,x.mean()+3*x.std()]
    
    
    print("Results of sklearn.metrics:")
    print("MAE:",mae)
    print("MAPE:",mape)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)
    print("3sigma:", threesigma_boundaries)
    print(x.mean(), str(x.std()))
    
    
    plt.hist(x, bins=150)
    plt.title("Histogram of Percentage Error")
    plt.xlabel("Percentage Error")
    plt.ylabel("Number of Occurances")
    plt.axvline(x.mean()-3*x.std(),  color="r", linestyle='--', label="Lower TSB")
    plt.axvline(x.mean()+3*x.std(), color="r",  linestyle='--' ,label="Upper TSB")
    plt.grid()
    plt.show()

    plt.show()
    
    with open("AccuracyCheckLogbook_2.csv", "a") as fileobj:
            fileobj.write(bestmodels[cntmodel]+","+str(mae)+","+str(mape)+","+str(mse)+","+str(rmse)+","+str(r2)+"\n")