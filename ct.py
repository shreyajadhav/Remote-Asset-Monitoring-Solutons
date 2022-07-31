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
P = solar_data.drop(['created_at','Current'], axis=1)
Q = solar_data[['created_at','Current']]
P_train, P_test, Q_train, Q_test = train_test_split(P, Q, test_size = 0.2, random_state = 2)
model2 = XGBRegressor()
model2.fit(P_train, Q_train.drop(['created_at'], axis=1))
training_data_predictionC = model2.predict(P_train)
test_data_predictionC = model2.predict(P_test)
C=pd.DataFrame(Q_test).reset_index(drop=True)
AC=pd.DataFrame(Q_test.drop(['created_at'],axis=1)).values.tolist()
PC=test_data_predictionC
from pandas.core.reshape.pivot import pivot
lst = Q_test['created_at']
plt.figure(figsize=(8,5))
plt.plot(lst, Q_test.drop(['created_at'],axis=1), 'b.-', label='Actual data')
plt.xticks(rotation=90)
plt.plot(lst, test_data_predictionC, 'g.-', label='Predicted data')
plt.xlabel('Time')
plt.ylabel('Current')
for i in range(len(AC)):
  if((AC[i]>PC[i]+2)|(AC[i]<PC[i]-2)):
    plt.scatter(C.loc[i]["created_at"], AC[i],c="red")
    plt.scatter(C.loc[i]["created_at"],PC[i],c="red")
plt.legend()
plt.show()
import time
for i in range(len(AC)):
  if((AC[i]>PC[i]+2)|(AC[i]<PC[i]-2)):
     print("Deviation at",C.loc[i]["created_at"])
     display(Audio(sound_file, autoplay=True))
     time.sleep(2)
  else:
    time.sleep(2)
