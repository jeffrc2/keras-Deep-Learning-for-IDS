import pandas
import math
import numpy

from sklearn.preprocessing import MinMaxScaler

train = pandas.read_csv('KDDTrain+processed.csv', header=None)

print(train.shape)

print(train)

print("Normalizing.")

labels = train[[train.columns[41]]]
y_train = labels.copy()

X_train = train.drop(train.columns[41],axis=1).copy()
Xscaler = MinMaxScaler(feature_range=(0,1))
Xscaler.fit(X_train)    
scaled_X_train = Xscaler.transform(X_train)
print(X_train.shape)
Yscaler = MinMaxScaler(feature_range=(0, 1))
Yscaler.fit(y_train)
scaled_y_train = Yscaler.transform(y_train)
print(scaled_y_train.shape)
scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)
print(scaled_y_train.shape)     

numpy.savetxt("X_train.csv", scaled_X_train, delimiter=", ")
numpy.savetxt("y_train.csv", scaled_y_train, delimiter=", ")

