# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:18:54 2020

@author: ogulc
"""

from time import process_time
import time
import torch
import torchvision
from torchvision import transforms, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import os
import scipy.stats as stats

 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

t0= process_time()






DF=pd.read_excel("MasterDB2019danberi_nomks_TOLD.xlsx")
DF = DF.sample(frac=1).reset_index(drop=True)
DF=DF[DF["DateasFDM"]<'2020-01-01']



altitudeparameterFP=['ParamA_FP',	'ParamB_FP','Altitude_MaxFP']
altitudeparameterFDM=['ParamA_FDM',	'ParamB_FDM','Altitude_MaxFDM']

distanceparameterFP=['AirDistanceFP',	'GroundDistanceFP','DurationFP']
distanceparameterFDM=['AirDistanceFDM',	'GroundDistanceFDM','DurationFDM']

def MAPEloss(output, target):
    loss=torch.mean(torch.abs((output - target))/torch.abs(target))                    
    return loss
  
for cntDF in range(len(altitudeparameterFP)):
    for cntDF2 in range(len(distanceparameterFP)):
        DFtrain=DF[["MachFP", 'TOW_FP', altitudeparameterFP[cntDF], distanceparameterFP[cntDF2], "FMdevFP",  "FuelBurnFP"]]
        DFtest=DF[["FID","MachFDM", 'TOW_FDM', altitudeparameterFDM[cntDF], distanceparameterFDM[cntDF2], "FMdevFDM", "FuelBurnFDM"]]
        
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        DFtrain=DFtrain.dropna()
        DFtest=DFtest.dropna()
        
        trainlen=int(len(DFtrain))
        testlen=int(len(DFtest)*.8)
        train00=DFtrain[:trainlen]
        valid00=DFtest[testlen:]
        test00=DFtest[:testlen]
        test00.to_excel("Test_Dataset/"+altitudeparameterFDM[cntDF]+"_"+distanceparameterFDM[cntDF2]+"_DF.xlsx")     
        valid00=valid00[["MachFDM", 'TOW_FDM', altitudeparameterFDM[cntDF], distanceparameterFDM[cntDF2], "FMdevFDM", "FuelBurnFDM"]]
        
        
        for cntnorm in range(len(train00.columns)):
            scaler = StandardScaler()
            train00[train00.columns[cntnorm]] = scaler.fit_transform(train00[train00.columns[cntnorm]].values.reshape(-1,1))
        for cntnorm1 in range(len(valid00.columns)):
            scaler1 = StandardScaler()
            valid00[valid00.columns[cntnorm1]] = scaler1.fit_transform(valid00[valid00.columns[cntnorm1]].values.reshape(-1,1))
        
        indepparams=5
        
                 
                                 
                             
        train0 = torch.tensor(train00.iloc[:,:indepparams].values.astype(np.float32),device=device)
        traintarget0 = torch.tensor(train00.iloc[:,indepparams:].values.astype(np.float32),device=device) 
        train = torch.utils.data.TensorDataset(train0, traintarget0)
        
                             
        valid0 = torch.tensor(valid00.iloc[:,:indepparams].values.astype(np.float32),device=device)
        validtarget0 = torch.tensor(valid00.iloc[:,indepparams:].values.astype(np.float32),device=device) 
        valid = torch.utils.data.TensorDataset(valid0, validtarget0)
        
        
        
        
        #test0 = torch.tensor(test00.iloc[:,:indepparams].values.astype(np.float32),device=device)
        #testtarget0 = torch.tensor(test00.iloc[:,indepparams:].values.astype(np.float32),device=device) 
        #test = torch.utils.data.TensorDataset(test0, testtarget0)
        
        trainset=torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
        validset=torch.utils.data.DataLoader(valid, batch_size=100, shuffle=True)
        #testset=torch.utils.data.DataLoader(test, batch_size=100, shuffle=True)
    
    
        
        staticnodearray=[16,32,64]
        
        
        
        class Net1(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1=nn.Linear(indepparams, staticnode)
                self.fc2=nn.Linear(staticnode,staticnode)
                self.fc3=nn.Linear(staticnode,staticnode)
                self.fc4=nn.Linear(staticnode,1)
                
            def forward(self, x):
                x=F.relu(self.fc1(x))
                x=F.relu(self.fc2(x))
                x=F.relu(self.fc3(x))
                x=self.fc4(x)
                
                return x
        
        
        
        class Net2(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1=nn.Linear(indepparams, staticnode)
                self.fc2=nn.Linear(staticnode,staticnode)
                self.fc3=nn.Linear(staticnode,1)
                
            def forward(self, x):
                x=F.relu(self.fc1(x))
                x=F.relu(self.fc2(x))
                x=self.fc3(x)
    
                return x
        
        
        
        lratearray=[0.01,0.001,0.0001]#
        epochsarray=[50,100,200]#
        
        for cntnode in range(len(staticnodearray)):
            staticnode=staticnodearray[cntnode]
            for cntlr in range(len(lratearray)):
                lrate=lratearray[cntlr]
                for cntepoch in range(len(epochsarray)):
                    epochs=epochsarray[cntepoch]
                    for cntmodelnet in range(2):
                        if cntmodelnet==0:
                           
                            net=Net1()
                            modellayer="4layers"
                        else:
                            net=Net2()
                            modellayer="3layers"
                        net=net.to(device)
                        optimizer=optim.Adam(net.parameters(), lr=lrate)
                    
                        
                        validloss_list = []
                        trainloss_list = []
                        for cntloss in range(2):
                            
                            for epoch in range(epochs):
                                epochloss=[]
                                for ts in trainset:
                                    X,y=ts
                                    net.zero_grad()
                                    output= net(X.view(-1,indepparams))
                                                      
                                    if cntloss==0:
                                        loss=MAPEloss(output, y)
                                        lossfunc="MAPE"
                                    else:
                                        loss=nn.MSELoss()
                                        loss=loss(output, y)
                                        lossfunc="MSE"
                                    epochloss.append(loss.item())
                                    loss.backward()
                                    optimizer.step()
                                        
                                trainloss_param=sum(epochloss)/len(epochloss)                       
                                
                                for vs in validset:
                                    X,y=vs      
                                    net.zero_grad()
                                    output= net(X.view(-1,indepparams))
                                   
                                    if cntloss==0:
                                        loss=MAPEloss(output, y)
                                        lossfunc="MAPE"
                                    else:
                                        loss=nn.MSELoss()
                                        loss=loss(output, y)
                                        lossfunc="MSE"
                                    
                                    epochloss.append(loss.item())
            
                                validloss_param=sum(epochloss)/len(epochloss)
                                validloss_list.append(validloss_param)
                                if validloss_param<=min(validloss_list):
                                    try:
                                        modelname='SavedModel_180422/'+str(altitudeparameterFP[cntDF])+"_"+str(distanceparameterFP[cntDF2])+"_"+str(staticnode)+"_"+str(lrate)+"_"+str(epochs)+"_"+modellayer+"_"+lossfunc+'.pth'
                                        torch.save(net.state_dict(), modelname)
                                        min_valid_loss=validloss_param
                                        inv_min_valid_loss=scaler1.inverse_transform([[min_valid_loss]])
                                        min_train_loss=trainloss_param
                                        inv_min_train_loss=scaler.inverse_transform([[min_train_loss]])
                                    except Exception as E:
                                        time.sleep(3)
                                        print(E)
                                        try:
                                            modelname='SavedModel_180422/'+str(altitudeparameterFP[cntDF])+"_"+str(distanceparameterFP[cntDF2])+"_"++str(staticnode)+"_"+str(lrate)+"_"+str(epochs)+"_"+modellayer+"_"+lossfunc+'.pth'
                                            torch.save(net.state_dict(), modelname)
                                            min_valid_loss=validloss_param
                                            inv_min_valid_loss=scaler1.inverse_transform([[min_valid_loss]])
                                            min_train_loss=trainloss_param
                                            inv_min_train_loss=scaler.inverse_transform([[min_train_loss]])
                                        except:
                                            with open("failed.txt", "a") as fileobj2:
                                                fileobj2.write(modelname+"\n")
                                            
                                    
                            
                            
                           
    
    
    
                                
                            with open("LOGS_180422.txt", "a") as fileobj:
                                fileobj.write(str(altitudeparameterFP[cntDF])+","+str(distanceparameterFP[cntDF2])+","+str(staticnode)+","+str(lrate)+","+str(epochs)+","+modellayer+","+lossfunc+","+
                                              str(min_train_loss)+","+str(min_valid_loss)+","+str(inv_min_train_loss)+","+str(inv_min_valid_loss)+"\n")
                                              
                        
            
t1 = process_time()
print("elapsed time on", device, t1-t0) 
                           
        
