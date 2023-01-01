# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:04:27 2021

@author: Engineering
"""

import pyodbc
import pandas as pd
from datetime import datetime, timedelta, date
 
conn36=pyodbc.connect('Driver={SQL Server};'
                              'Server=localhost;'
                              'Database=FuelMasterDB;'
                              'Trusted_Connection=yes;'
                              )

FP=pd.read_sql_query("Select * from FlightPlans Where Date>'2022-07-25'", con=conn36)
FDM=pd.read_sql_query("Select * from [FDM State Values] where [A C TO Datetime]>'2022-07-25'", con=conn36)
APM=pd.read_sql_query("Select * from APM Where [Datee]>'2022-07-25'", con=conn36)
Mach=pd.read_sql_query("Select Mach, KeyWYP from Waypoints Where [Date]>'2022-07-25' AND Mach<>'' and Mach IS NOT NULl and Mach<>'null'", con=conn36)
LevelParam=pd.read_excel("leveldata27092022.xlsx")    #leveloutput.py




FP["Dateas"]=pd.to_datetime(FP["Date"], format='%Y-%m-%d')
FP["DateasFP"]=FP["Dateas"].dt.date
FP["keytoFDM"]=FP["DateasFP"].astype(str)+"_"+FP["DepartureAirport"]+"_"+FP["ArrivalAirport"]+"_"+FP["Aircraft"]
FP = FP.drop_duplicates(subset=['keytoFDM'],keep='last')

FDM["Dateas"]=pd.to_datetime(FDM["A C TO Datetime"], format='%Y-%m-%d')
FDM["DateasFDM"]=FDM["Dateas"].dt.date
FDM["aircraft_reg"]=FDM["aircraft_reg"].str.replace("-","", regex=True)
FDM["keytoFP"]=FDM["DateasFDM"].astype(str)+"_"+FDM["TO Airport"]+"_"+FDM["TD Airport"]+"_"+FDM["aircraft_reg"]
FDM = FDM.drop_duplicates(subset=['keytoFP'],keep='last')
MergedDF=pd.merge(left=FDM, right=FP, left_on='keytoFP', right_on='keytoFDM')



APM["Aircraft"]=APM["Aircraft"].str.replace("-","", regex=True)
APM["dep-dest"]=APM["Dept Dest"].str[0:4]+"_"+APM["Dept Dest"].str[4:8]
APM["FM Dev (%)"]=pd.to_numeric(APM["FM Dev (%)"], downcast="float")
APM["APMkey"]=APM["Datee"].astype(str)+"_"+APM["dep-dest"]+"_"+APM["Aircraft"]
APM=APM[["APMkey", "FM Dev (%)"]]
APM=APM.groupby(['APMkey']).mean()
MergedDF=pd.merge(left=MergedDF, right=APM, left_on='keytoFDM', right_on='APMkey')


MergedDF["FID"]=MergedDF['FID'].astype(int)
LevelParam["FID"]=LevelParam['FID'].astype(int)
MergedDF=pd.merge(left=MergedDF, right=LevelParam, left_on='FID', right_on='FID')


Mach["MachFP"]=pd.to_numeric(Mach["Mach"], errors='coerce', downcast="float")
Mach=Mach.groupby(['KeyWYP']).mean()
MergedDF=pd.merge(left=MergedDF, right=Mach, left_on='KeyFP', right_on='KeyWYP')



MergedDF=MergedDF.apply(pd.to_numeric, errors='ignore' )



MergedDF['MachFDM']=MergedDF['Mach - Cruise (mean)']

MergedDF["FBcalc0"]=MergedDF["Fuel Burn - B Takeoff"] +MergedDF["Fuel Burn - H Landing"] +MergedDF["Fuel Burn - C Initial Climb"] + MergedDF["Fuel Burn - D Climb"]+ MergedDF["Fuel Burn - E Cruise"]+MergedDF["Fuel Burn - F Descent"] + MergedDF["Fuel Burn - G Approach"] 
MergedDF["FBcalc1"]=MergedDF["Fuel Burn - Cruise"]+MergedDF[  "Fuel Burn - Climb"]+MergedDF[  "Fuel Burn - Descent"]
MergedDF['FuelBurnFDM']=MergedDF[["FBcalc0",  "FBcalc1"]].max(axis=1)


MergedDF["ADcalc0"]=  MergedDF["Air Distance - C Initial Climb"] + MergedDF["Air Distance - D Climb"]+ MergedDF["Air Distance - E Cruise"]+MergedDF["Air Distance - F Descent"] + MergedDF["Air Distance - G Approach"]
MergedDF["ADcalc1"]=MergedDF["Air Distance - Cruise"]+MergedDF[  "Air Distance - Climb"]+MergedDF[  "Air Distance - Descent"]
MergedDF['AirDistanceFDM']=MergedDF[["ADcalc0",  "ADcalc1"]].max(axis=1)


MergedDF["GDcalc0"]= MergedDF["Ground Distance - B Takeoff"] + MergedDF["Ground Distance - H Landing"]+ MergedDF["Ground Distance - C Initial Climb"] + MergedDF["Ground Distance - D Climb"]+ MergedDF["Ground Distance - E Cruise"]+MergedDF["Ground Distance - F Descent"] + MergedDF["Ground Distance - G Approach"] 
MergedDF["GDcalc1"]=MergedDF["Ground Distance - Cruise"]+MergedDF[  "Ground Distance - Climb"]+MergedDF[  "Ground Distance - Descent"]
MergedDF['GroundDistanceFDM']=MergedDF[["GDcalc0",  "GDcalc1"]].max(axis=1)

MergedDF["Durcalc0"]= MergedDF["Duration - B Takeoff"] + MergedDF["Duration - H Landing"]+MergedDF["Duration - C Initial Climb"] + MergedDF["Duration - D Climb"]+ MergedDF["Duration - E Cruise"]+MergedDF["Duration - F Descent"] + MergedDF["Duration - G Approach"] 
MergedDF["Durcalc1"]=MergedDF["Duration - Cruise"]+MergedDF[  "Duration - Climb"]+MergedDF[  "Duration - Descent"]
MergedDF['DurationFDM']=MergedDF[["Durcalc0",  "Durcalc1"]].max(axis=1)

MergedDF['TOW_FDM']=MergedDF['FuelBurnFDM']+  MergedDF['Weight - H Landing']

MergedDF['FMdevFDM']=MergedDF['FM Dev (%)']

MergedDF['Altitude_MaxFDM']=MergedDF['Altitude - E Cruise (max)']



MergedDF[['Hour','Minute',"Sec"]] = MergedDF["TripFuelDuration"].str.split(':',expand=True)
MergedDF["DurationFP"]=MergedDF['Hour'].astype(int)*60+MergedDF['Minute'].astype(int)
MergedDF['FuelBurnFP']=MergedDF['TripFuelWeight']
MergedDF['GroundDistanceFP']=MergedDF['GroundDistance']
MergedDF['AirDistanceFP']=MergedDF['AirDistance']
MergedDF['TOW_FP']= MergedDF['TakeoffWeight']

MergedDF['FMdevFP']=(MergedDF['CruiseDegradation']-1)*(-100)
MergedDF['Altitude_MaxFP']=MergedDF['FlightLevel']*100

MergedDF=MergedDF[MergedDF['aircraft_reg']!='TCMKS']








MergedDF=MergedDF[['FID','aircraft_reg','DateasFDM',  'FuelBurnFDM','MachFDM','GroundDistanceFDM', 'AirDistanceFDM',"DurationFDM", 'TOW_FDM','FMdevFDM', 'ParamA_FDM', 'ParamB_FDM', 'Altitude_MaxFDM',
                   'FuelBurnFP','MachFP','GroundDistanceFP','AirDistanceFP',"DurationFP",'TOW_FP','FMdevFP', 'ParamA_FP', 'ParamB_FP', 'Altitude_MaxFP']]



MergedDF.to_excel("MasterDB27092022.xlsx")
