# Fuel-Consumption-Estimation-Medium-Fidelity-FFNN
A feed-forward neural network for fuel consumption estimation in a commercial flight


These scripts have been used in a master thesis that can be accessed in the below link;

https://open.metu.edu.tr/handle/11511/99592

The ANN model has been trained by the flight flan data, validated and tested by the flight data. 

leveloutput.py parametrizes altitude data over a flight profile. 
DB_creation.py merges all the necessary databases and creates a output database. 
ANN model - Hypertuning.py tunes hyperparameters and logs all the models. 
StatisticalAccuracyCheck_.py compares the model results. 

sample_dataset.xlsx is the sample dataset to train, validate and test the data. 
