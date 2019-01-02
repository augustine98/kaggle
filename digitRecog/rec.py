import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

x_train= df[0:100]
x_train.to_csv('digitHead.csv',index=False)

x1= x_train[1]
print(x1)
#print (len(df))
#print(len(x_train))
plt.imshow(x1)


