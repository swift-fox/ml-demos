import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

N = 50  # Number of words to generate

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", use_cache=True)

# Input a sentence
print("Type something for the model:")
input_string = input()

# Tokenize the input and initialize variables
inputs = tokenizer(input_string, return_tensors="tf")["input_ids"]
past = None
outputs = []

# Generate the output
for _ in range(N):
    logits, past = model(inputs, past)
    next_id = tf.argmax(logits[:, -1, :], 1)  # Greedy search: always select the next word with the largest probability
    outputs.append(next_id[0].numpy())
    inputs = next_id[tf.newaxis, :]

# Translate tokens back to words
print(tokenizer.decode(outputs))
