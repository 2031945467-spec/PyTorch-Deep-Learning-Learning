import torch
import torch.nn as nn
from fastapi import FastAPI

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,3)
        self.conv2=nn.Conv2d(16,32,3)
        self.pool=nn.MaxPool2d(2,2)

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
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

app = FastAPI()
@app.get("/")
def home():
    return {"info": "MNIST 真实手写数字识别"}

@app.get("/predict")
def predict():
    img=torch.randn(1,1,28,28)
    with torch.no_grad():
        output=model(img)
        pred=torch.argmax(output,dim=1).item()
    return {"预测数字": pred}
