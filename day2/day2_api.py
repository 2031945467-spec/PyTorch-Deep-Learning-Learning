import torch
import torch.nn as nn
from fastapi import FastAPI
#定义模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.flatten(x)
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x
#加载模型
model=Model()
model.load_state_dict(torch.load("day2_model.pth"))
model.eval()
#启动服务
app=FastAPI()
@app.get("/")
def home():
    return {"message":"Day2 服务启动成功"}
@app.get("/predict")
def predict():
    img=torch.randn((1,1,28,28))
    with torch.no_grad():
        out=model(img)
        pred=torch.argmax(out).item()
    return {"pred":pred}