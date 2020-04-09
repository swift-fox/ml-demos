import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import Sequential, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, ReLU, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def resize(item):
    image, mask = item['image'], item['segmentation_mask']
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 127.5 - 1
    mask = tf.image.resize(mask, (128, 128))
    mask -= 1
    return image, mask

# Data pipeline
train, valid = tfds.load('oxford_iiit_pet', split=['train', 'test'])

train = train.map(resize).shuffle(1000).batch(64)
valid = valid.map(resize).batch(64)

# Construct the feature extraction path from the base vision model
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False)
base_model.trainable = False

feature_layers = [
    'block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu',
    'block_13_expand_relu', 'block_16_project'
]
features = [base_model.get_layer(name).output for name in feature_layers]
feature_model = Model(inputs=base_model.input, outputs=features)

# Construct the up-sampling path of using Conv2DTranspose with strides 2
inputs = Input(shape=(128, 128, 3))
feature_maps = feature_model(inputs)
x = feature_maps[-1]

filters_size = [512, 256, 128, 64]
feature_maps = feature_maps[-2::-1]

for size, map in zip(filters_size, feature_maps):
    x = Conv2DTranspose(size, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = tf.concat([x, map], -1)    # Build the skip connection

output = Conv2DTranspose(3, 3, strides=2, padding='same')(x)
model = Model(inputs=inputs, outputs=output)

# Train it!
model.compile('adam', SparseCategoricalCrossentropy(True), ['accuracy'])
model.fit(train, validation_data=valid, epochs=5)

# Save the model
model.save('models/oxford_iiit_pet_unet.h5')
