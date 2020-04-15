import matplotlib.pyplot as plt
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

# Vision model based on ResNet
base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1),
])

# Train the last Dense layer
base_model.trainable = False
model.compile(Adam(1e-3), BinaryCrossentropy(True), ['accuracy'])
model.fit(train, validation_data=valid)

# Fine-tuning
base_model.trainable = True
model.compile(Adam(1e-4), BinaryCrossentropy(True), ['accuracy'])
model.fit(train, validation_data=valid, epochs=4)

# Save the model
model.save('models/cats_vs_dogs')

# Generate the heatmap
image = next(iter(train.unbatch().take(1)))[0]

heatmap = base_model(image[tf.newaxis, :])[0]
heatmap = tf.reduce_mean(heatmap, -1)

plt.imshow(image)
plt.imshow(heatmap, alpha=0.6, extent=(0, 160, 160, 0), interpolation='bilinear')
plt.show()
