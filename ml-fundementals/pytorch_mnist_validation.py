import torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

validate_set = datasets.MNIST('data', download=True, train=False, transform=transform)
validate_loader = torch.utils.data.DataLoader(validate_set, shuffle=True)

model = torch.load('mnist.pt')

all_count = 0
correct_count = 0

for image, label in validate_loader:
    image = image.view(1, 784)

    with torch.no_grad():
        pred = model(image)

    pred = list(torch.exp(pred).numpy()[0])
    pred_label = pred.index(max(pred))
    true_label = label.item()

    if true_label == pred_label:
      correct_count += 1
    all_count += 1

print("all = {}, correct = {}, {}%".format(all_count, correct_count, float(correct_count) / all_count * 100))
