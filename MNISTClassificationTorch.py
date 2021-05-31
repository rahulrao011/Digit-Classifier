import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test  =datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

model = nn.Sequential()
model.add_module('0', nn.Linear(28*28, 64))
model.add_module('1', nn.ReLU())
model.add_module('2', nn.Linear(64, 32))
model.add_module('3', nn.ReLU())
model.add_module('4', nn.Linear(32, 16))
model.add_module('5', nn.ReLU())
model.add_module('6', nn.Linear(16, 10))
model.add_module('7', nn.LogSoftmax(dim=1))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 3

for i in range(EPOCHS):
    for data in trainset:
        X, y = data
        model.zero_grad()
        output = model(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = model(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print(f'Accuracy: {round(correct/total, 3)}')
print('-'*20,'DONE','-'*20)