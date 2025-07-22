import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
x=np.linspace(-1, 1, 100)
y=2*x + 1 + np.random.randn(100) 
print(x)
print(y)
x.reshape(-1, 1)  
print(x.shape)
model= Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=100)
pred=model.predict(x)
plt.scatter(x, y, label='original data')
plt.plot(x, pred, label='predicted data')
plt.show()