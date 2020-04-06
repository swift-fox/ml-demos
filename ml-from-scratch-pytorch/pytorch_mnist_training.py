import torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST('data', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

I = 784
H1 = 128
H2 = 64
O = 10

model = nn.Sequential(
    nn.Linear(I, H1),
    nn.ReLU(),
    nn.Linear(H1, H2),
    nn.ReLU(),
    nn.Linear(H2, O),
    nn.LogSoftmax(dim=1)
)

loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

for e in range(15):
    _loss = 0
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)

        pred = model(images)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _loss += loss.item()

    print("Epoch {} - Training loss: {}".format(e, _loss/len(train_loader)))

torch.save(model, "mnist.pt")
