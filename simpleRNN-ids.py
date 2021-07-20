import tensorflow
import pandas as pd
import numpy as np
import os
from time import time
from tensorflow import keras
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical

X_file = open("X_train.csv")

trainX = np.loadtxt(X_file, delimiter=",")



y_file = open("y_train.csv")

trainY = np.loadtxt(y_file, delimiter=",")

trainY = to_categorical(trainY, num_classes=5)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

X_file = open("X_test.csv");
testX = np.loadtxt(X_file, delimiter=",")

y_file = open("y_test.csv");
testY = np.loadtxt(y_file, delimiter=",")

testY = to_categorical(testY, num_classes=5)

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = keras.Sequential()

model.add(layers.SimpleRNN(units=80, input_dim=41, activation="sigmoid", return_sequences=False))

model.add(layers.Dropout(0.1))

model.add(layers.Dense(5,activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="rnn-ids-checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True,monitor='val_accuracy',mode='max')
csv_logger = CSVLogger('training_set_iranalysis.csv',separator=',', append=False)

model.summary()

model.fit(trainX, trainY, batch_size=32, epochs=100, validation_data=(testX, testY), callbacks=[checkpointer,csv_logger])

model.save("rnnIDS_model.hdf5")

loss, accuracy = model.evaluate(testX, testY)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_pred = model.predict_classes(testX)

np.savetxt('rnnIDSpredicted.txt', np.transpose([testY,y_pred]), fmt='%01d')

# model.predict(testX)