import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, LSTM, RNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Data pipeline
(train, valid), info = tfds.load('imdb_reviews/subwords8k', split=['train', 'test'],
    as_supervised=True, with_info=True)

vocab_size = info.features['text'].encoder.vocab_size

train = train.shuffle(1000).padded_batch(128, padded_shapes=([None],[]))
valid = valid.padded_batch(128, padded_shapes=([None],[]))

# Embedding + LSTM model
model = Sequential([
    Embedding(vocab_size, 16),
    Bidirectional(LSTM(64)),
    Dense(1),
])

# Train it!
model.compile(Adam(0.01), BinaryCrossentropy(True), ['accuracy'])
model.fit(train, validation_data=valid, epochs=5)

# Save the model
model.save('models/imdb_subwords8k_lstm')
