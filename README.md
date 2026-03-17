# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

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
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sameeha S
RegisterNumber: 212225230243 
*/
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
df = pd.read_csv("Salary.csv")
print("Dataset Preview:")
print(df.head())
X = df[["Level"]]  
y = df["Salary"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeRegressor(criterion="squared_error",max_depth=3,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mse)
print("RMSE :", rmse)
print("R2   :", r2_score(y_test, y_pred))
plt.figure(figsize=(16, 10))
plot_tree(model,feature_names=["Level"],filled=True)
plt.title("Decision Tree Regressor for Employee Salary Prediction")
plt.show()
new_exp = [[5]]  
predicted_salary = model.predict(new_exp)
print("\nPredicted Salary for 5 years experience:", predicted_salary[0])
```

## Output:

<img width="1246" height="250" alt="image" src="https://github.com/user-attachments/assets/cab9c0e0-e0a2-4884-975c-01b61a6b2cfd" />
<img width="1366" height="752" alt="image" src="https://github.com/user-attachments/assets/e3da12e7-b6b7-4904-861f-d5ac4d384e0e" />
<img width="1295" height="135" alt="image" src="https://github.com/user-attachments/assets/59bd503b-d666-4c94-a4c5-10277d9bf147" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
