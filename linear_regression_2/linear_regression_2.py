# -*- coding: utf-8 -*-

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset=load_diabetes()

data_train,data_test,target_train,target_test=train_test_split(dataset.data
                                                              ,dataset.target)
                                                             

regression=LinearRegression()
regression.fit(data_train,target_train)

print(regression)

# Denklem=ax+b
# b------
print('Doğrunun Başlangıç Noktası:',regression.intercept_)
# a------
print('Katsayılar:',regression.coef_)

y_predictions=regression.predict(data_test)
print('Doğruyu Oluşturan Noktalar:',y_predictions)