import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
print(df.shape)
print(df.count())

fig = plt.figure(figsize=(36,24))
plt.subplot2grid((2,3),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
#plt.legend(())
plt.title("Survived Distribution")

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived,df.Age,alpha =0.2)
plt.title("Age vs Survived")


plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Normalized Pclass")


plt.subplot2grid((2,3),(1,0),colspan=2)
for x in [1,2,3]:
    df.Age[df.Pclass==x].plot(kind="kde")
plt.title("Class Density vs Age")
plt.legend(("1st","2nd","3rd"))




plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts(normalize=True).plot(kind="bar")

plt.savefig('titanic.png')
plt.show()
