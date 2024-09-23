# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Import the standard Libraries.

Step 2. Set variables for assigning dataset values.

Step 3. Import linear regression from sklearn.

Step 4. Assign the points for representing in the graph.

Step 5. Predict the regression for marks by using the representation of the graph.

Step 6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Shashank R 
RegisterNumber: 212223230205
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
![image](https://github.com/user-attachments/assets/f8b94176-f7c8-444b-b67b-b1b679af64f7)
![image](https://github.com/user-attachments/assets/a7a3802b-d4b1-48d1-9526-6bc97d784b87)
![image](https://github.com/user-attachments/assets/2ea20f5f-8455-4037-8999-3edada69a9e2)
![image](https://github.com/user-attachments/assets/a2b3f7c0-4e83-47fe-b9f5-82db5b8131c2)
![image](https://github.com/user-attachments/assets/adc5f0d6-c0ba-4cc5-9ce1-3bc13a5faf13)
![image](https://github.com/user-attachments/assets/76da8792-b4dc-47aa-b6f6-34b4630d1980)
![image](https://github.com/user-attachments/assets/b68c688b-35c6-4181-8c87-1dc8fba8f604)
![image](https://github.com/user-attachments/assets/7385981a-dacc-4989-909a-d1012e9599be)
![image](https://github.com/user-attachments/assets/19831476-0423-4764-83af-0d4da74804af)
![image](https://github.com/user-attachments/assets/21aa32c3-29f6-4e10-973b-d13cad4a928b)
![image](https://github.com/user-attachments/assets/92c19276-8378-44d1-8acf-3b5ee00e01da)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
