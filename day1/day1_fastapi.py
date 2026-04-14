# 导包
import torch
import torch.nn as nn
from fastapi import FastAPI


# 定义模型\
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


# 加载模型
model = Model()
model.load_state_dict(torch.load("day1_model.pth"))
model.eval()
# 启动服务
app = FastAPI()


@app.get("/")
def home():
    return {"message": "启动成功"}


@app.get("/predict")
def predict(x1: float, x2: float):
    data = torch.tensor([[x1, x2]])
    out = model(data).item()
    return out
