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
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=train_transform)
train_set,val_set=random_split(full_dataset,[50000,10000])
test_set=datasets.MNIST(root='./data',train=False,download=False,transform=test_transform)

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
loss_fn=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=1e-3)

epochs = 8
best_val_acc = 0
patience = 3
early_stop_count = 0

print("开始训练...")
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
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels=images.to(device),labels.to(device)
            pred=model(images)
            val_correct+=(torch.argmax(pred,dim=1)==labels).sum().item()
            val_total=val_total+labels.size(0)
    val_acc=val_correct/val_total
    print(f"Epoch {epoch + 1} | 训练损失: {train_loss / len(train_loader):.3f} | 验证准确率: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc=val_acc
        torch.save(model.state_dict(), "best_mnist_cnn.pth")
        print(f"✅ 验证准确率提升，保存最优模型（当前最高：{best_val_acc:.3f}）")
        early_stop_count = 0
    else:
        early_stop_count+=1
        print(f"⚠️  验证无提升，早停计数：{early_stop_count}/{patience}")
        if early_stop_count >= patience:
            print("🛑 触发早停，提前结束训练")
            break