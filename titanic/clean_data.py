def clean(data):
    data.loc[data["Sex"]=="male","Sex"]=0
    data.loc[data["Sex"]=="female","Sex"]=1

    data.loc[data["Embarked"]=="S","Embarked"]=0
    data.loc[data["Embarked"]=="C","Embarked"]=1
    data.loc[data["Embarked"]=="Q","Embarked"]=2
    

