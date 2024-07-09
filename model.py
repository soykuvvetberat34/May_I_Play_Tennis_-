import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
data=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\çoklu doğrusal regresyon\\odev.csv")
outlook=data.iloc[:,0].values.reshape(-1,1)
windy=data.iloc[:,3].values.reshape(-1,1).ravel()
play=data.iloc[:,4].values.reshape(-1,1).ravel()
temp=data.iloc[:,1].values.reshape(-1,1).ravel()
hum=data.iloc[:,2].values.reshape(-1,1).ravel()
temp=pd.Series(temp)
hum=pd.Series(hum)
#outlook OneHot encoding
OHE=OneHotEncoder()
OHE_outlook=OHE.fit_transform(outlook).toarray()
outlook_df=pd.DataFrame(OHE_outlook,columns=["sunny","overcast","rainy"])
#windy encoding
LE=LabelEncoder()
windy_LE=LE.fit_transform(windy)
windy_LE=pd.Series(windy_LE)
#play encoding
play_LE=LE.fit_transform(play)
play_LE=pd.Series(play_LE)
result_1=pd.concat([outlook_df,temp],axis=1)
result_2=pd.concat([result_1,hum],axis=1)
result_3=pd.concat([result_2,windy_LE],axis=1)
result_3=result_3.iloc[:,[0,1,2]].values
print(result_3)
#backward elemination
array=np.append(arr=np.ones((14,1)).astype(int),values=result_3,axis=1)
model=sm.OLS(play_LE,result_3).fit()
print(model.summary())
df=pd.DataFrame(result_3,columns=["outlook","temperature","humidity"])
print(df)
x_train,x_test,y_train,y_test=train_test_split(result_3,play_LE,test_size=0.33,random_state=1)
LRegression=LinearRegression()
LRegression.fit(x_train,y_train)
predict=LRegression.predict(x_test)
print(y_test)
y_test=np.array(y_test)
print(predict)

for i in range(5):
    print(f"Doğru(True) değer {y_test[i]} \n tahmin(predict) edilen değer {predict[i]}\n")






