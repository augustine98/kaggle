import pandas as pd
import uti
import csv
from sklearn import linear_model,preprocessing


with open('train.csv','r') as f1:
    csv_reader = csv.reader(f1)
    for row in csv_reader:
        for i, x in enumerate(row):
                if len(x)< 1:
                        row[i] = 0


df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')

feature_names=["Pclass","Sex","Age","Fare","SibSp","Parch","Embarked"]

df = df.reset_index()
uti.clean(df)
target =df.Survived.values
features = df[feature_names].values


uti.clean(dt)
dt =dt.reset_index()
test_feat=dt[feature_names].values




classifier=linear_model.LogisticRegression()
#classifier_=classifier.fit(features,target)

#print(classifier_.score(features,target))


poly=preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

poly_classifier_=classifier.fit(poly_features,target)
print(poly_classifier_.score(poly_features,target))




dt["Survived"]=0
test_poly_features = poly.fit_transform(test_feat)

yPredict= classifier.predict(test_poly_features)
dt.Survived = yPredict

res = dt[["PassengerId","Survived"]]
print (res)

res.to_csv("poly_logistic_submission.csv",index=False)
#print(yPredict)





#print(classifier_.score(test_feat))








