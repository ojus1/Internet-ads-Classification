import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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

X = np.delete(df,1558,1)
Y = df[:,1558]
Y = Y.reshape(3279,1)

Scores = list()

for i in range(0,50) :
    X_tr, X_ts, Y_tr, Y_ts = train_test_split(X,Y)

    model = SVC()
    model.fit(X_tr,Y_tr)
    Scores.append(model.score(X_ts,Y_ts))

plt.plot(Scores)
plt.show()