# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:04:17 2021

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
query="Select * from Level Where (FID=?) and (Category='Altitude- FP' or Category='Altitude- FDM')"
FIDdb=pd.read_sql_query("Select Distinct FID from Level Where InstantTime>'2022-07-25'", con=conn36)
FIDdb=FIDdb.applymap(str)

df2w=pd.DataFrame([], columns=("FID", "ParamA_FDM", "ParamB_FDM", "ParamA_FP", "ParamB_FP") )
for cnt1 in range(len(FIDdb)):
    
    param={FIDdb.iat[cnt1,0]}
    cnt5=0
    try:
        LVL=pd.read_sql_query(query, con=conn36, params=param)
        
        AltFDM=LVL[LVL["Category"]=="Altitude- FDM"]
        AltFDM=AltFDM.sort_values(by='InstantTime')
        AltFDM=AltFDM.reset_index()
        cnt2=0
        LPFDM=0
        totalcruiseduration=0
        levelparamA=0
        levelparamB=0
        totalduration=((AltFDM.iat[-1,2]-AltFDM.iat[0,2]).total_seconds())/60
        
       
        while cnt2<len(AltFDM)-1:
            tpa=AltFDM.iat[cnt2+1,2]
            ta=AltFDM.iat[cnt2,2]
            lpa=AltFDM.iat[cnt2+1,4]
            la=AltFDM.iat[cnt2,4]
            
            
            duration=((tpa-ta).total_seconds())/60
            if lpa==la:
                dummylevelparamA=lpa*duration
                totalcruiseduration=totalcruiseduration+duration
                levelparamA=levelparamA+dummylevelparamA
                dummylevelparamB=lpa*duration
                levelparamB=levelparamB+dummylevelparamB
            
            else:
                dummylevelparamB=(lpa+la)*duration/2
                levelparamB=levelparamB+dummylevelparamB
                
            cnt2+=1
        
        ParamA_FDM=levelparamA*totalduration/(totalcruiseduration**2)   
        ParamB_FDM=levelparamB/totalduration
            
      
        
        AltFP=LVL[LVL["Category"]=="Altitude- FP"]
        AltFP=AltFP.sort_values(by='InstantTime')
        AltFP=AltFP.reset_index()
        totalcruiseduration=0
        levelparamA=0
        levelparamB=0
        totalduration=((AltFP.iat[-1,2]-AltFP.iat[0,2]).total_seconds())/60
        cnt3=0
        
       
        while cnt3<len(AltFP)-1:
            tpa=AltFP.iat[cnt3+1,2]
            ta=AltFP.iat[cnt3,2]
            lpa=AltFP.iat[cnt3+1,4]
            la=AltFP.iat[cnt3,4]
            
            
            duration=((tpa-ta).total_seconds())/60
            if lpa==la:
                dummylevelparamA=lpa*duration
                totalcruiseduration=totalcruiseduration+duration
                levelparamA=levelparamA+dummylevelparamA
                dummylevelparamB=lpa*duration
                levelparamB=levelparamB+dummylevelparamB
            
            else:
                dummylevelparamB=(lpa+la)*duration/2
                levelparamB=levelparamB+dummylevelparamB
            cnt3+=1
        ParamA_FP=levelparamA*totalduration/(totalcruiseduration**2)   
        ParamB_FP=levelparamB/totalduration
        
       
        dummyDF=pd.DataFrame({"FID":[FIDdb.iat[cnt1,0]],"ParamA_FDM": [ParamA_FDM], "ParamB_FDM": [ParamB_FDM],"ParamA_FP": [ParamA_FP], "ParamB_FP": [ParamB_FP]})
        
       
    
        df2w=pd.concat([df2w, dummyDF])
    except:
        cnt5+=1
        pass
   

df2w.to_excel("leveldata27092022.xlsx")
    