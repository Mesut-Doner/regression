# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


dataset=load_diabetes()
x_data=dataset.data[:,np.newaxis,2]  # Yeni boyut

x_train=x_data[:-50]
x_test=x_data[-50:]

y_train=dataset.target[:-50]
y_test=dataset.target[-50:]

regression=LinearRegression()
regression.fit(x_train,y_train)

# Denklem=ax+b
# b------
print('Doğrunun Başlangıç Noktası:',regression.intercept_)
# a------
print('Katsayılar:',regression.coef_)

y_predictions=regression.predict(x_test)
print('Doğruyu Oluşturan Noktalar',y_predictions)


plt.plot(x_test,y_predictions,color='red',linewidth=3)
plt.scatter(x_train,y_train,color='green')
plt.scatter(x_test,y_predictions,color='blue')