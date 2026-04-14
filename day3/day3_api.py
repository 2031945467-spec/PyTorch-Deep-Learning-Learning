import torch
import torch.nn as nn
from fastapi import FastAPI


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Model()
model.load_state_dict(torch.load("day3_model.pth"))
model.eval()

app = FastAPI()


@app.get("/")
def home():
    return {"info": "Day3 数字识别服务"}


@app.get("/predict")
def predict():
    img = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, dim=1).item()
    return {"pred":pred}