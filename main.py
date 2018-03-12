import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

f = open('ads_data.txt','r')
lines = f.readlines()
f.close()
lines = [line.replace(' ', '') for line in lines]
f = open('cleaned_ads_data.txt','w')
f.writelines(lines)
f.close()

df = pd.read_csv('cleaned_ads_data.txt', header= None)
df = df.replace('?',np.nan)
df = pd.get_dummies(df, columns= [1558], drop_first= True)
imp = Imputer(strategy= 'mean', axis=0)
df = imp.fit_transform(df)

X = df.drop(columns= [1558])
Y = df[[1558]]
