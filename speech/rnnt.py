import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, Softmax
from tensorflow.keras.optimizers import Adam

from warprnnt_tensorflow import rnnt_loss

def preprocess(x, y):
    # TODO: audio processing, convert to mel spectrogram, etc
    return x, y

# Data pipeline
(train, valid), info = tfds.load('librispeech/subwords8k', split=['train_clean100', 'test_clean'], with_info=True)

vocab_size = info.features['text'].encoder.vocab_size

train = train.map(preprocess).shuffle(100).padded_batch(10, padded_shapes=([None],[]))
valid = valid.map(preprocess).padded_batch(10, padded_shapes=([None],[]))

vocab_size = 8192

# The model
encoder_net = Sequential([
    Bidirectional(LSTM(128)),
    Dense(128),
])

prediction_net = Sequential([
    Embedding(vocab_size, 128),
    LSTM(128),
    Dense(128),
])

joint_net = Sequential([
    Dense(128),
    Dense(vocab_size, 'tanh'),
    Softmax(),
])

x = Input(shape=(1024, 128))
y = Input(shape=(1,))

h_enc = encoder_net(x)
h_pre = prediction_net(y)
p = joint_net(tf.concat([h_enc, h_pre], -1))

model = Model(inputs=[x, y], outputs=p)

# Train it!
optim = Adam(1e-4)
train_loss = tf.keras.metrics.Mean(name='train_loss')

for epoch in range(20):
    train_loss.reset_states()

    for batch in train:
        with tf.GradientTape() as tape:
            pred = model(batch['x'], batch['y'])
            loss = rnnt_loss(y, pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)

# Save the model
model.save('models/librispeech_subwords8k_rnnt')
