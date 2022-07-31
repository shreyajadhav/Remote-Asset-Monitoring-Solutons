import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import IPython
sound_file = '/content/mixkit-classic-short-alarm-993.wav'
from IPython.display import Audio, display
solar_dataset = pd.read_csv('SolarData.csv')
solar_data = solar_dataset.drop(['entry_id','latitude','longitude','elevation','status'], axis=1)
X = solar_data.drop(['created_at','Voltage'], axis=1)
Y = solar_data[['created_at','Voltage']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
model = XGBRegressor()
model.fit(X_train, Y_train.drop(['created_at'], axis=1))
training_data_prediction = model.predict(X_train)
test_data_prediction = model.predict(X_test)
AV=pd.DataFrame(Y_test.drop(['created_at'],axis=1)).values.tolist()
V=pd.DataFrame(Y_test).reset_index(drop=True)
PV=test_data_prediction
from pandas.core.reshape.pivot import pivot
lst = Y_test['created_at']
plt.figure(figsize=(8,5))
plt.plot(lst, Y_test.drop(['created_at'], axis=1), 'b.-', label='Actual data')
plt.xticks(rotation=90)
plt.plot(lst, test_data_prediction, 'g.-', label='Predicted data')
plt.xlabel('Time')
plt.ylabel('Voltage')
for i in range(len(AV)):
  if((AV[i]>PV[i]+2)|(AV[i]<PV[i]-2)):
    plt.scatter(V.loc[i]["created_at"],AV[i],c="red")
    plt.scatter(V.loc[i]["created_at"],PV[i],c="red")

plt.legend()
plt.show()
import time
for i in range(len(AV)):
  if((AV[i]>PV[i]+2)|(AV[i]<PV[i]-2)):
     print("Deviation at",V.loc[i]["created_at"])
     display(Audio(sound_file, autoplay=True))
     time.sleep(2)
  else:
    time.sleep(2)
