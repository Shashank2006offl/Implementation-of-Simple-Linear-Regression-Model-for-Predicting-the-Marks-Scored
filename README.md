# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored



## AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


Step 1: Start program

Step 2. Import the standard Libraries.

Step 3. Set variables for assigning dataset values.

Step 4. Import linear regression from sklearn.

Step 5. Assign the points for representing in the graph.

Step 6. Predict the regression for marks by using the representation of the graph.

Step 7. Compare the graphs and hence we obtained the linear regression for the given datas.

Step 8. End Program.


## Program:


```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Shashank R
RegisterNumber:  212222240027
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
#load dataset
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```



## Output:


Values of Y prediction

![ypred](https://user-images.githubusercontent.com/119393424/229979053-f32194cb-7ed4-4326-8a39-fe8186079b63.png)


Array values of Y test

![ytest](https://user-images.githubusercontent.com/119393424/229979114-3667c4b7-7610-4175-9532-5538b83957ac.png)


Array value of X:![xvalue](https://user-images.githubusercontent.com/119393424/229978918-707c006d-0a30-4833-bf77-edd37e8849bb.png)


Array value of Y

![yvalue](https://user-images.githubusercontent.com/119393424/229978994-b0d2c87c-bef9-4efe-bba2-0bc57d292d20.png)


Training Set Graph: ![image](https://github.com/user-attachments/assets/eddb4993-677e-4b01-9c5a-ff363088a261)


Test Set Graph :

![test](https://user-images.githubusercontent.com/119393424/229979225-ba90853c-7fe0-4fb2-8454-a6a0b921bdc1.png)


Values of MSE, MAE and RMSE: ![mse](https://user-images.githubusercontent.com/119393424/229979276-bb9ffc68-25f8-42fe-9f2a-d7187753aa1c.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
