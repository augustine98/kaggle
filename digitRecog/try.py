import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np



#d= pd.read_csv('digitHead.csv').as_matrix()
X= pd.read_csv('train.csv').as_matrix()
T= pd.read_csv('test.csv').as_matrix()



clf = tree.DecisionTreeClassifier()

X_train=X[:,1:]
Y_train =X[:,0]
#X_test=X[21001:,1:]
#Y_test =X[21001:,0]



X_test2=T

clf.fit(X_train,Y_train)
#p =clf.predict(X_test)
#eq =(p==Y_test)
#print(np.sum(eq)/len(eq))


Yt = clf.predict(X_test2)

df = pd.DataFrame(Yt,columns=["Label"])
df["ImageId"]= 0
for i in range (0,len(df)):
    df["ImageId"][i]=i+1

df = df[["ImageId","Label"]]
df.to_csv('submit.csv',index=False)




