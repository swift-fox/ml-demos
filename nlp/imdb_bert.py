import bert, os
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

MAX_LEN = 512

# Get BERT and tokenizer
bert_model_dir = bert.fetch_google_bert_model("uncased_L-12_H-768_A-12", "bert_models")
bert_model_ckpt = os.path.join(bert_model_dir, "bert_model.ckpt")
bert_params = bert.params_from_pretrained_ckpt(bert_model_dir)

vocab_file = os.path.join(bert_model_dir, "vocab.txt")
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, True)

# Data pipeline
train, valid = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True)

def tokenize(text, label):
    def _tokenize(text, label):
        tokens = tokenizer.tokenize(text.numpy())[:MAX_LEN - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids, label
    return tf.py_function(_tokenize, [text, label], [tf.int32, tf.int64])

train = train.map(tokenize).padded_batch(128, padded_shapes=([MAX_LEN],[]))
valid = valid.map(tokenize).padded_batch(128, padded_shapes=([MAX_LEN],[]))

# Construct a classifier with BERT
bert_layer = bert.BertModelLayer.from_params(bert_params)
bert_layer.trainable = False

model = Sequential([
    Input(shape=(MAX_LEN,)),
    bert_layer,
    Lambda(lambda seq: seq[:, 0, :]),
    Dense(1),
])

bert.load_bert_weights(bert_layer, bert_model_ckpt)

# Train it!
model.compile(Adam(), BinaryCrossentropy(True), ['accuracy'])
model.fit(train, validation_data=valid, epochs=5)

# Save the model
model.save('models/imdb_bert')
