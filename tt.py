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
A = solar_data.drop(['created_at','Temp'], axis=1)
B = solar_data[['created_at','Temp']]
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.2, random_state = 2)
model1 = XGBRegressor()
model1.fit(A_train, B_train.drop(['created_at'], axis=1))
training_data_predictionT = model1.predict(A_train)
test_data_predictionT = model1.predict(A_test)
AT=pd.DataFrame(B_test.drop(['created_at'],axis=1)).values.tolist()
T=pd.DataFrame(B_test).reset_index(drop=True)
PT=test_data_predictionT
from pandas.core.reshape.pivot import pivot
lst =B_test['created_at']
plt.figure(figsize=(10,6))
plt.plot(lst, B_test.drop(['created_at'],axis=1), 'b.-', label='Actual data')
plt.xticks(rotation=90)
plt.plot(lst, test_data_predictionT, 'g.-', label='Predicted data')
plt.xlabel('Time')
plt.ylabel('Temperature')
for i in range(len(AT)):
  if((AT[i]>PT[i]+5)|(AT[i]<PT[i]-5)):
    plt.scatter(T.loc[i]["created_at"],AT[i],c="red")
    plt.scatter(T.loc[i]["created_at"],PT[i],c="red")
plt.legend()
plt.show()
import time
for i in range(len(AT)):
  if((AT[i]>PT[i]+5)|(AT[i]<PT[i]-5)):
     print("Deviation at",T.loc[i]["created_at"])
     display(Audio(sound_file, autoplay=True))
     time.sleep(2)
  else:
    time.sleep(2)
