# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 18:01:55 2018

@author: Augustine
"""
import matplotlib.pyplot as plt
x=[1,2,3]
y=[2,4,6]
i =1
while i<=100:
	for t in range (0,len(x)):
		y[t]=i*x[t]
	plt.plot(x,y)
	i+=1
plt.show()
	
