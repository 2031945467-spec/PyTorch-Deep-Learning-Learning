import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
train_transform=transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

full_train=datasets.MNIST(root='./data',train=True,download=True,transform=train_transform)
train_set,val_set=random_split(full_train,[50000,10000])
test_set=datasets.MNIST(root='./data',train=False,transform=test_transform)

train_loader=DataLoader(train_set,batch_size=64,shuffle=True)
val_loader=DataLoader(val_set,batch_size=64,shuffle=False)
test_loader=DataLoader(test_set,batch_size=64,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,3)
        self.conv2=nn.Conv2d(16,32,3)
        self.pool=nn.MaxPool2d(2,2)
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(32*5*5,128)
        self.drop=nn.Dropout(0.5)
        self.fc2=nn.Linear(128,10)

    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.flatten(x)
        x=torch.relu(self.fc1(x))
        x=self.drop(x)
        x=self.fc2(x)
        return x

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=1e-3)

epochs=10
for epoch in range(epochs):
    model.train()
    train_loss=0
    for images,labels in train_loader:
        images,labels=images.to(device),labels.to(device)
        pred=model(images)
        loss=loss_fn(pred,labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss+=loss.item()

    model.eval()
    val_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels=images.to(device),labels.to(device)
            pred=model(images)
            val_loss+=loss_fn(pred,labels).item()

            pre_label=torch.argmax(pred,dim=1)
            correct+=(pre_label==labels).sum().item()
            total+=labels.size(0)

    print(f"Epoch {epoch + 1}")
    print(f"训练损失：{train_loss / len(train_loader):.3f}")
    print(f"验证损失：{val_loss / len(val_loader):.3f}")
    print(f"验证准确率：{correct / total:.3f}")
    print("-" * 50)