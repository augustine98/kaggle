import pandas as pd

df= pd.read_csv('train.csv')

df["Hyp"]=0
#Simply predict survival if passenger is female
df.loc[df.Sex=="female","Hyp"]=1

df["Res"]=0

df.loc[df.Survived==df["Hyp"],"Res"]=1

print (df["Res"].value_counts(normalize=True))