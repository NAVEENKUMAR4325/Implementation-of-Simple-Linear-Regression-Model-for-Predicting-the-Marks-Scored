# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Naveen Kumar.E
RegisterNumber:212222220029
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv('/content/ML_02_data.csv')
df.head()
df.tail()
#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#displaying predicted values
y_pred
#displaying actual values
y_test
#graph plot for training data
plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
## df.head()

![image](https://github.com/NAVEENKUMAR4325/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119479566/ba28fdd3-21a3-4fae-ba8d-ab06cad588e7)

## df.tail()

![image](https://github.com/NAVEENKUMAR4325/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119479566/42965d1d-1091-44d9-88ff-d8a832a9d851)

## Array value of X:

![image](https://github.com/NAVEENKUMAR4325/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119479566/6841b119-aea5-47c8-9520-6937cea7ed60)

## Array value of Y:

![image](https://github.com/NAVEENKUMAR4325/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119479566/807f6b48-74fc-433e-81e9-ad0d905ad807)

## Value of Y prediction:

![image](https://github.com/NAVEENKUMAR4325/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119479566/52c4d677-866a-4753-8ed1-389d8c93d7e7)

## Values of Y test:

![image](https://github.com/NAVEENKUMAR4325/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119479566/a1cbc489-1340-4c1d-8bc7-2d1bc1d75b47)

## Training Set Graph and Test Set Graph:

![image](https://github.com/NAVEENKUMAR4325/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119479566/f54547f6-3d01-413b-8a27-0b47ff45681f)

## Values of MSE, MAE and RMSE:

![image](https://github.com/NAVEENKUMAR4325/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119479566/442a6109-e71d-442e-963d-0b2a27c57f61)











## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
