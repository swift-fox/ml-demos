import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dropout, Flatten, Dense, Softmax
from tensorflow.keras.losses import SparseCategoricalCrossentropy

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10)#, activation='softmax')
])

model.compile('adam', SparseCategoricalCrossentropy(True), ['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

final_model = Sequential([
    model,
    Softmax()
])

pred = final_model(x_test[:5])
print(pred)
