import pandas as pd
from sklearn import tree,model_selection,preprocessing
import uti

dtr =pd.read_csv('train.csv')
dtes=pd.read_csv('test.csv')


dtr=uti.clean(dtr)
dtes=uti.clean(dtes)

target = dtr["Survived"].values

feature_names=["Pclass","Sex","Age","Fare","SibSp","Parch","Embarked"]
output_features=["PassengerId","Survived"]

features = dtr[feature_names]
test_feat=dtes[feature_names]

# polynomial features for decision tree
poly=preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)
poly_test_features =poly.fit_transform(test_feat)


"""
decision_tree= tree.DecisionTreeClassifier(random_state=1)
decision_tree_= decision_tree.fit(features,target)
print(decision_tree_.score(features,target))

scores= model_selection.cross_val_score(decision_tree,features,target,scoring='accuracy',cv=50)
print (scores)
print(scores.mean())

"""

generalized_tree=tree.DecisionTreeClassifier(random_state=1,max_depth=15,min_samples_split=2)
generalized_tree_=generalized_tree.fit(poly_features,target)

scores =model_selection.cross_val_score(generalized_tree,poly_features,target,scoring="accuracy",cv=100)
print(scores)
print(scores.mean())



dtes["Survived"]= generalized_tree.predict(poly_test_features)
#print(dtes.Survived)
res = dtes[output_features]
res.to_csv("generalized_poly_decision_tree_submission3.csv",index=False)






