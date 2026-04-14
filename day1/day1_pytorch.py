# 导包
import torch
import torch.nn as nn
import torch.optim as optim

# 准备数据
x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32)
y = torch.tensor([[0], [0], [1], [1]], dtype=torch.float32)


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

#初始化模型 损失函数 优化器
model = Model()
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.01)
# 训练模型
for epoch in range(100):
    pred=model(x)
    loss=loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# 保存模型
torch.save(model.state_dict(), "day1_model.pth")
