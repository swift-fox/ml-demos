import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images, test_images = train_images / 255.0, test_images / 255.0

model = Sequential([
    Conv2D(32, 3, activation='relu'),
    MaxPool2D(),
    Conv2D(32, 3, activation='relu'),
    MaxPool2D(),
    Conv2D(32, 3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10),
])

model.compile('adam', SparseCategoricalCrossentropy(True), ['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
