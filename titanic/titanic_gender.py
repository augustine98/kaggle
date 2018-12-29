import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
print(df.shape)
print(df.count())

fig = plt.figure(figsize=(36,24))
plt.subplot2grid((3,4),(0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Survived")

plt.subplot2grid((3,4),(0,1))
df.Survived[df.Sex=="male"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Men Survived")


plt.subplot2grid((3,4),(0,2))
df.Survived[df.Sex=="female"].value_counts(normalize=True).plot(kind="bar",alpha=0.5)
plt.title("Women Survived")


plt.subplot2grid((3,4),(0,3))
df.Sex[df.Survived==1].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color=['r','b'])
plt.title("Sex Survived")



plt.subplot2grid((3,4),(1,0), colspan=4)
for x in [1,2,3]:
    df.Survived[df.Pclass==x].plot(kind="kde",alpha=0.6)
plt.legend(("1st", "2nd","3rd"))
plt.title("Class Density vs Survived")

plt.subplot2grid((3,4),(2,0))
df.Survived[(df.Pclass==1)&(df.Sex=="male")].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color='r')
plt.title("1st Class Men Survived")

plt.subplot2grid((3,4),(2,1))
df.Survived[(df.Pclass==3)&(df.Sex=="male")].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color='r')
plt.title("3rd Class Men Survived")

plt.subplot2grid((3,4),(2,2))
df.Survived[(df.Pclass==1)&(df.Sex=="female")].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color='b')
plt.title("1st Class Women Survived")

plt.subplot2grid((3,4),(2,3))
df.Survived[(df.Pclass==3)&(df.Sex=="female")].value_counts(normalize=True).plot(kind="bar",alpha=0.5,color='b')
plt.title("3rd Class Woen Survived")



plt.savefig('titanic_gender.png')
plt.show()