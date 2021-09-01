# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

dataset=load_diabetes()
x_data=dataset.data[:,np.newaxis,2]  # Yeni boyut

x_train=x_data[:-15]
x_test=x_data[-15:]

y_train=dataset.target[:-15]
y_test=dataset.target[-15:]


regression=MLPRegressor(max_iter=1000)
regression.fit(x_train,y_train)

y_predictions=regression.predict(x_test)
print(y_predictions)

plt.plot(x_test,y_predictions,color='red',linewidth=3)
plt.scatter(x_train,y_train,color='green')
plt.scatter(x_test,y_predictions,color='blue')