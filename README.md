# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber: 212224230175 
*/
```
    import pandas as pd
    data=pd.read_csv("Placement_Data.csv")
    data.head()
    
    data1=data.copy()
    data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
    data1.head()
    
    data1.isnull().sum()
    
    data1.duplicated().sum()
    
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    data1["gender"]=le.fit_transform(data1["gender"])
    data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
    data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
    data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
    data1["degree_t"]=le.fit_transform(data1["degree_t"])
    data1["workex"]=le.fit_transform(data1["workex"])
    data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
    data1["status"]=le.fit_transform(data1["status"])       
    data1 
    
    x=data1.iloc[:,:-1]
    x
    y=data1["status"]
    y
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression(solver="liblinear")
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    y_pred
    
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    accuracy
    
    from sklearn.metrics import confusion_matrix
    confusion=confusion_matrix(y_test,y_pred)
    confusion
    
    from sklearn.metrics import classification_report
    classification_report1 = classification_report(y_test,y_pred)
    print(classification_report1)
    lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
    
## Output:

<img width="1259" height="255" alt="Screenshot_2025-09-27_094307 1" src="https://github.com/user-attachments/assets/6edff456-9647-40b8-b8bf-1a352aa22169" />
<img width="1407" height="269" alt="Screenshot_2025-09-27_094255 1" src="https://github.com/user-attachments/assets/31a3b2fd-6cce-4ff1-8184-de66a62bc269" />
<img width="223" height="57" alt="Screenshot_2025-09-27_094322 1" src="https://github.com/user-attachments/assets/65b61601-d04d-4b0a-b032-814607b4c11d" />
<img width="296" height="611" alt="Screenshot_2025-09-27_094315 1" src="https://github.com/user-attachments/assets/93a0bf9c-7734-4baa-b58e-d0a2b2c6ece1" />
<img width="1154" height="530" alt="Screenshot_2025-09-27_094332 1" src="https://github.com/user-attachments/assets/4fd1496c-5650-4e6b-90d9-7796d372194f" />
<img width="275" height="565" alt="Screenshot_2025-09-27_094340 1" src="https://github.com/user-attachments/assets/3138f6d8-e58c-45d1-ab07-ef31aaabb56c" />
<img width="747" height="69" alt="Screenshot_2025-09-27_094349 1" src="https://github.com/user-attachments/assets/576f6d20-f76f-4dfa-a391-f46b73068be8" />
<img width="268" height="40" alt="Screenshot_2025-09-27_094355 1" src="https://github.com/user-attachments/assets/e5dbae94-a090-4b81-a754-60f34303bcab" />
<img width="297" height="61" alt="Screenshot_2025-09-27_094402 1" src="https://github.com/user-attachments/assets/d312bdd6-608d-46d0-9ea2-e76f59975a21" />
<img width="674" height="233" alt="Screenshot_2025-09-27_094409 1" src="https://github.com/user-attachments/assets/279f2ebc-6b1d-48c0-9eda-c15b08ea2d94" />
<img width="379" height="43" alt="Screenshot_2025-09-27_094233 1" src="https://github.com/user-attachments/assets/3d415e72-95c2-4cd3-b59a-a2b0ce84fd71" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
