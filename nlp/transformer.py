import torch
import torch.nn.functional as F
from torch import nn

def scaled_dot_product_attention(query, key, value, mask):
    scores = query.bmm(key.transpose(1, 2))
    scores /= key.size(-1) ** 0.5

    if mask is not None:
        scores += mask

    softmax = F.softmax(scores, dim=-1)
    return softmax.bmm(value)

class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query, key, value, mask):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        return scaled_dot_product_attention(q, k, v, mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_k, dim_v):
        super().__init__()
        heads = [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        self.heads = nn.ModuleList(heads)
        self.liner = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query, key, value, mask=None):
        x = [head(query, key, value, mask) for head in self.heads]
        x = torch.cat(x, dim=-1)
        x = self.liner(x)
        return x

def positional_encoding(seq_len, dim_model, device):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim / dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input, dim_feedforward):
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class Residual(nn.Module):
    def __init__(self, sublayer, dim, dropout):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors, **kwargs):
        x = self.sublayer(*tensors, **kwargs)
        x = self.dropout(x)
        x += tensors[-1]
        x = self.norm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, dim_model, dim_feedforward, dropout):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dim_model, dropout
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dim_model, dropout
        )

    def forward(self, x):
        x = self.attention(x, x, x)
        x = self.feed_forward(x)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, dim_model, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        seq_len, dimension = x.size(1), x.size(2)
        x += positional_encoding(seq_len, dimension, device=x.device)
        for layer in self.layers:
            x = layer(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, num_heads, dim_model, dim_feedforward, dropout):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dim_model, dropout
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dim_model, dropout
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dim_model, dropout
        )

    def forward(self, y, enc, mask):
        y = self.attention_1(y, y, y, mask=mask)
        y = self.attention_2(y, enc, enc)
        y = self.feed_forward(y)
        return y

class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, dim_model, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(num_heads, dim_model, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(dim_model, vocab_size)

    def forward(self, y, enc):
        seq_len, dimension = y.size(1), y.size(2)
        y += positional_encoding(seq_len, dimension, device=y.device)

        mask = torch.full((seq_len, seq_len), float('-inf'), device=y.device).triu(1)

        for layer in self.layers:
            y = layer(y, enc, mask)

        y = self.linear(y)
        y = torch.softmax(y, dim=-1)
        return y

class Transformer(nn.Module):
    def __init__(self, vocab_size = 40000, num_encoder_layers = 6, num_decoder_layers = 6,
                dim_model = 512, num_heads = 8, dim_feedforward = 2048, dropout = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.encoder = Encoder(num_encoder_layers, num_heads, dim_model, dim_feedforward, dropout)
        self.decoder = Decoder(num_decoder_layers, num_heads, vocab_size, dim_model, dim_feedforward, dropout)

    def forward(self, x, y):
        x = self.embedding(x)
        y = self.embedding(y)

        enc = self.encoder(x)
        return self.decoder(y, enc)

if __name__ == '__main__':
    import torchtext
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    specials = ['<unk>', '<bos>', '<eos>']

    # Load the dataset and create the tokenizer
    train = torchtext.datasets.WikiText2(split='train')
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train), specials=specials)
    vocab.set_default_index(vocab['<unk>'])

    unk, bos, eos = vocab(specials)
    vocab_size = len(vocab)

    # Data pipeline
    def generate_data(seq):
        x = torch.tensor(seq + [eos], device=device)
        y = torch.tensor([bos] + seq, device=device)
        return x, y

    def preprocess(dataset):
        dataset = map(lambda seq: vocab(tokenizer(seq)), dataset)
        dataset = filter(lambda seq: len(seq) > 0, dataset)
        dataset = list(map(generate_data, dataset))
        return DataLoader(dataset, batch_size=1, shuffle=True)

    train = torchtext.datasets.WikiText2(split='train')
    train = preprocess(train)

    # Create a model
    model = Transformer(vocab_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # Train it!
    for epoch in range(5):
        _loss = 0.0
        for x, y in train:
            pred = model(x, y)
            loss = loss_fn(pred.squeeze(), x.squeeze())
            _loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch {}, loss {} '.format(epoch, _loss))

    # Save the vocab and model
    torch.save(vocab, 'vocab.pt')
    torch.save(model, 'transformer.pt')
