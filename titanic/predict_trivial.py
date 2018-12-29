import pandas as pd

df= pd.read_csv('test.csv')

df["Survived"]=0
#Simply predict survival if passenger is female
df.loc[df.Sex=="female","Survived"]=1

df.loc[df.Pclass==1,"Survived"]=1

#df["Res"]=0

ds = df[["PassengerId","Survived"]]

ds.to_csv('trivial.csv',index=False)

#df.loc[df.Survived==df["Hyp"],"Res"]=1

