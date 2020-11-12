import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

N = 50  # Number of words to generate
n_beams = 3 # Number of beams in beam search

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", use_cache=True)

# Input a sentence
print("Type something for the model:")
input_string = input()

# Tokenize the input and initialize variables
inputs = tokenizer(input_string, return_tensors="tf")["input_ids"]
past = None
history = [None] * n_beams
priors = [[1.0]]

# Generate the output
for _ in range(N):
    logits, past = model(inputs, past)
    probs = tf.nn.softmax(logits[:, -1, :]) * priors

    probs = tf.reshape(probs, -1)
    priors, indices = tf.math.top_k(probs, n_beams, False)   # Beam search: always select top n words with largest probabilities

    inputs = indices % tokenizer.vocab_size
    beam_idx = indices // tokenizer.vocab_size

    history = [(id.numpy(), history[prev]) for id, prev in zip(inputs, beam_idx)]
    past = [tf.gather(layer_past, beam_idx, axis=1) for layer_past in past]

    inputs = inputs[:, tf.newaxis]
    priors = priors[:, tf.newaxis]

# Reconstructe the output from the end
outputs = []
prev = history[tf.argmax(priors)[0]]

while prev:
    id, prev = prev
    outputs.append(id)

outputs.reverse()

# Translate tokens back to words
print(tokenizer.decode(outputs))
