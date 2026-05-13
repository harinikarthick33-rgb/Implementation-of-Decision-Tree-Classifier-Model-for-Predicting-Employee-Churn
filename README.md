# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1. Import libraries and load dataset.
2. Convert categorical data to numeric values.
3. Split data into training and testing sets.
4. Train the Decision Tree model.
5. Predict test results.
6. Calculate accuracy.
7. Display the decision tree.


## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
data = pd.read_csv(r"C:\Users\acer\Downloads\Employee .csv")
data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harini V K

RegisterNumber:  212225220036
*/


## Output:
<img width="1395" height="687" alt="image" src="https://github.com/user-attachments/assets/8011b903-48eb-4254-b3e0-da2190acbfb5" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
