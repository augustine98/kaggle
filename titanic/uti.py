def clean(data,test):

    data["Fare"]=data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"]=data["Age"].fillna(data["Age"].dropna().median())
    data["Pclass"]=data["Pclass"].fillna(data["Pclass"].dropna().median())
    
    
    data.loc[data["Sex"]=="male","Sex"]=0
    data.loc[data["Sex"]=="female","Sex"]=1

    data["Embarked"]=data["Embarked"].fillna("S")
    data.loc[data["Embarked"]=="S","Embarked"]=0
    data.loc[data["Embarked"]=="C","Embarked"]=1
    data.loc[data["Embarked"]=="Q","Embarked"]=2


    #data=data.drop(['PassengerId'], axis=1)

    #Chopping off everything excepth the title
    data["Name_Title"] = data["Name"].apply(lambda x : x.split(',')[1]).apply(lambda x: x.split()[0])

    title ={"Mr.": 0 , "Mrs.":1,"Miss.":2 ,"Master.":3}

    for i in range(0,len(data["Name_Title"])):
        if data["Name_Title"][i] not in title:
                data["Name_Title"][i] ="Rare"
    
    title["Rare"]= 4


    data.Name_Title = data.Name_Title.ma(title)




    data=data.drop(['Name'],axis=1)
    data=data.drop(['Ticket'],axis=1)
    data=data.drop(['Cabin'],axis=1)



    data["Age_Class"]=data["Age"]*data["Pclass"]
    data["Relatives"] = data["SibSp"]+data["Parch"]
    data["Alone"]=1
    data.loc[data['Relatives'] > 0, 'Alone'] = 0


    data['Fare_Per_Person'] = data['Fare']/(data['Relatives']+1)
    data['Fare_Per_Person'] = data['Fare_Per_Person'].astype(int)


    pId = data[["PassengerId"]]
    X=data.drop(["PassengerId"],axis=1)

    if(not test):
        y=data[["Survived"]]
        X=X.drop(["Survived"],axis=1)
        return pId,X,y
    else :
        return pId,X


    


    

