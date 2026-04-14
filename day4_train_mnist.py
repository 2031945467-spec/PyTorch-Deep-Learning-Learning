import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([transforms.ToTensor(),])

train_set=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_set=datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader=DataLoader(train_set,batch_size=64,shuffle=True)
test_loader=DataLoader(test_set,batch_size=64,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,32,3)

        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(32*5*5,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.flatten(x)
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x

model=CNN()
loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=0.001)

model.train()
for epoch in range(10):
    total_loss=0
    for images,labels in train_loader:
        pred=model(images)
        loss=loss_fn(pred,labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(train_loader)
    print(f"Epoch {epoch + 1} | Loss: {avg_loss:.3f}")

model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        output=model(images)
        prediction=torch.argmax(output,dim=1)
        correct+=(prediction==labels).sum().item()
        total+=labels.size(0)

acc=correct/total
print(f"\n真实测试集准确率: {acc*100:.2f}%")

torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型已保存：mnist_cnn.pth")