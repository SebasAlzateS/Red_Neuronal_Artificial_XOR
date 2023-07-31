# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 08:21:34 2023

@author: Usuario
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
# y estos son los resultados que se obtienen, en el mismo orden
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(32, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='relu')) #1 capa de salida
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])


model.fit(training_data, target_data, epochs=1000)

salida= model.predict(training_data)
error= np.sqrt(np.sum(np.square(target_data-salida)))/4
print("El error es: ", error)
print(salida)

model.save('modeloprueba.h5')


