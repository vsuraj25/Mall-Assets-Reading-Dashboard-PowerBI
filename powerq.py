import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def factorize_LMH(data):
    if data == 'Low':
        return 0
    elif data == 'Medium':
        return 1
    elif data == 'High':
        return 2 
    
dataset = dataset[['Month','Outdoor_Temperature','Indoor_Temperature','Occupancy','Fan_Speed','Air_Filter','Air_Quality_PPM','Power_consumption_kwh']]
dataset['Fan_Speed'] = dataset['Fan_Speed'].apply(factorize_LMH)
dataset['Air_Filter'] = dataset['Air_Filter'].apply(factorize_LMH)
y = dataset.iloc[:,-1]
x = dataset.iloc[:, :-1]
scalar =  StandardScaler()
x = scalar.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.25)
rfr =  RandomForestRegressor()
rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_test)
dataset['Prediction'] = rfr.predict(x)