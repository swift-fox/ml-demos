import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def resize(image, label):
    image = tf.image.resize(image, (160, 160))
    image = tf.cast(image, tf.float32) / 255
    return image, label

# Data pipeline
train, valid = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], as_supervised=True)

train = train.map(resize).cache().shuffle(1000).batch(100)
valid = valid.map(resize).cache().batch(100)

# Transer learning model
model = Sequential([
    ResNet50V2(include_top=False, weights='imagenet'),
    GlobalAveragePooling2D(),
    Dense(1),
])

# Training the last Dense layer
model.layers[0].trainable = False
model.compile(Adam(1e-3), BinaryCrossentropy(True), ['accuracy'])
model.fit(train, validation_data=valid) # Yes, one epoch is enough

# Fine tuning
model.layers[0].trainable = True
model.compile(Adam(1e-4), BinaryCrossentropy(True), ['accuracy'])
model.fit(train, validation_data=valid, epochs=4)

# Save the model
model.save('models/cats_vs_dogs.h5')
