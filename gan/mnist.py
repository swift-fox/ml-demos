import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, ReLU, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Flatten, Reshape

batch_size = 256
latent_dim = 100

# Data pipeline
(images, _), _ = tf.keras.datasets.mnist.load_data()
images = images / 255.0
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)

# The generator model
generator = Sequential([
    Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)),
    BatchNormalization(),
    ReLU(),
    Reshape((7, 7, 256)),
    Conv2DTranspose(128, 5, 1, 'same', use_bias=False),
    BatchNormalization(),
    ReLU(),
    Conv2DTranspose(64, 5, 2, 'same', use_bias=False),
    BatchNormalization(),
    ReLU(),
    Conv2DTranspose(1, 5, 2, 'same', use_bias=False, activation='tanh'),
])

# The discriminator model
discriminator = Sequential([
    Conv2D(64, 5, 2, 'same', input_shape=(28, 28, 1)),
    ReLU(),
    Dropout(0.3),
    Conv2D(128, 5, 2, 'same'),
    ReLU(),
    Dropout(0.3),
    Flatten(),
    Dense(1),
])

loss_fn = tf.keras.losses.BinaryCrossentropy(True)
gen_optim = tf.keras.optimizers.Adam(1e-4)
disc_optim = tf.keras.optimizers.Adam(1e-4)
gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')

# Train them!
for epoch in range(50):
    for real_images in dataset:
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(noise, training=True)

            real_pred = discriminator(real_images, training=True)
            fake_pred = discriminator(fake_images, training=True)

            gen_loss = loss_fn(tf.ones_like(fake_pred), fake_pred)
            disc_loss = loss_fn(tf.ones_like(real_pred), real_pred) + loss_fn(tf.zeros_like(fake_pred), fake_pred)

        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optim.apply_gradients(zip(gen_grads, generator.trainable_variables))

        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optim.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        gen_loss_metric(gen_loss)
        disc_loss_metric(disc_loss)

    print(f'epoch {epoch + 1}, gen_loss {gen_loss_metric.result():.3f}, disc_loss {disc_loss_metric.result():.3f}')
    gen_loss_metric.reset_states()
    disc_loss_metric.reset_states()

# Save the models
generator.save('models/mnist_gan_generator.h5')
discriminator.save('models/mnist_gan_discriminator.h5')

# Generate some fake images
noise = tf.random.normal([16, latent_dim])
generated_images = generator(noise, training=False)

fig = plt.figure(figsize=(4,4))

for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i, :, :, 0] * 255.0, cmap='gray')
    plt.axis('off')

plt.show()
