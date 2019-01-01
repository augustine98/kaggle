import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uti
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


X_tr = pd.read_csv('train.csv')
X_ts = pd.read_csv('test.csv')


#X_tr,X_ts,a,b = train_test_split(X,y,test_size=0.3,random_state=42)


pId_tr, X_train,y_train =uti.clean(X_tr,False)
pId_ts, X_test =uti.clean(X_ts,True)

print( " Train Size ", X_train.shape)
print( " Train Size ", y_train.shape)
print( " Test Size ", X_test.shape)
print(X_train.head())
print(X_test.head())
#print( " Train Size ", y_test.shape)



#model =DecisionTreeClassifier()

model = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

"""
print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())
"""

print(predictions)

#scor = [predictions==y_test]  

#print(np.sum(sum(scor)))
"""
cnt= 0
for i in range(0,len(predictions)):
    if predictions[i]==y_test["Survived"][i]:
        cnt+=1

print("Accuracy ",cnt/len(predictions))
"""

df2 =pd.DataFrame(predictions,columns=["Survived"])
df2["PassengerId"]=pId_ts
out = df2[["PassengerId","Survived"]]
out.to_csv("final.csv",index=False)

print(df2)
