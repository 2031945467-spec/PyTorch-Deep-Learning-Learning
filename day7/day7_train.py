import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

test_set = datasets.MNIST('../data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16,3)
        self.conv2 = nn.Conv2d(16, 32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*5*5,128)
        self.drop=nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.flatten(x)
        x=torch.relu(self.fc1(x))
        x=self.drop(x)
        x = self.fc2(x)
        return x

model = CNN().to(device)
model.load_state_dict(torch.load('../models/best_mnist_cnn.pth'))
model.eval()

with torch.no_grad():
    images,labels=next(iter(test_loader))
    images,labels=images.to(device),labels.to(device)
    outputs=model(images)
    preds=torch.argmax(outputs,dim=1)

images,labels,preds=images.cpu().numpy(),labels.cpu().numpy(),preds.cpu().numpy()

plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(images[i].squeeze(),cmap='gray')
    color='blue' if preds[i]==labels[i] else 'red'
    plt.title(f'T:{labels[i]} P:{preds[i]}',color=color)
plt.tight_layout()
plt.show()
