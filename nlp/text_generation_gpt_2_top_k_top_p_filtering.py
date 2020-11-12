import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

N = 50  # Number of words to generate
k = 10   # Top-k items to select from
p = 0.8 # Top-p cumulative probability to select from

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
k = tokenizer.vocab_size if not k else k

# Generate the output
for _ in range(N):
    logits, past = model(inputs, past)

    # Top-k filtering: select only k items with largest probabilities
    logits, indices = tf.math.top_k(logits[:, -1, :], k)

    # Top-p filtering: select only top items within cumulative probability of p
    cumsum = tf.math.cumsum(tf.nn.softmax(logits), 1)
    selected = cumsum <= max(p, cumsum[0][0])   # Make sure at least 1 item is selected
    logits, indices = logits[selected], indices[selected]

    _inf = tf.fill([tokenizer.vocab_size], -float('inf'))
    logits = tf.tensor_scatter_nd_update(_inf, indices[:,tf.newaxis], logits)

    inputs = tf.random.categorical(logits[tf.newaxis,:], 1)  # Sample the next word
    outputs.append(inputs[0][0].numpy())

# Translate tokens back to words
print(tokenizer.decode(outputs))
