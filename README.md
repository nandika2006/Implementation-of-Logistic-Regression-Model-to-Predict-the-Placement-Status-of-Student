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
<img width="1221" height="226" alt="image" src="https://github.com/user-attachments/assets/d806499f-3f8e-4817-b085-858b0e1bb427" />
<img width="1091" height="240" alt="image" src="https://github.com/user-attachments/assets/eb768478-75db-4288-94de-429a957b6c4e" />
<img width="982" height="497" alt="image" src="https://github.com/user-attachments/assets/4771a6ca-8ec1-4d03-86c6-9da514777799" />

    Data Duplicate:
<img width="61" height="48" alt="image" src="https://github.com/user-attachments/assets/8a7c70e1-e4da-4cdc-aae9-474aaf0b8ab5" />

    Print data:
<img width="982" height="502" alt="image" src="https://github.com/user-attachments/assets/4aef18f6-a3cb-4763-aa6a-c79a5856627b" />

    Data-Status:
<img width="922" height="510" alt="image" src="https://github.com/user-attachments/assets/37ec3566-d7cb-4fc2-b12e-39e1256cf7f2" />

    y_prediction array:
<img width="586" height="263" alt="image" src="https://github.com/user-attachments/assets/86ba868e-9ede-4e1e-9872-88d7ebe50082" />

    Confusion array:
<img width="762" height="71" alt="image" src="https://github.com/user-attachments/assets/84a84927-5a7f-4d1b-9c56-75cf81fb071b" />

    Accuracy Value:
<img width="210" height="51" alt="image" src="https://github.com/user-attachments/assets/4d820e47-9986-4e2f-9cb8-637a2e1ae4cf" />

    Classification Report:
<img width="582" height="176" alt="image" src="https://github.com/user-attachments/assets/b1b9d6c7-ac8a-4991-998a-00b6bcb5a616" />

    Prediction of LR:
<img width="303" height="33" alt="image" src="https://github.com/user-attachments/assets/f1a29b97-8440-48dd-b9f9-7ad0da8938b2" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
