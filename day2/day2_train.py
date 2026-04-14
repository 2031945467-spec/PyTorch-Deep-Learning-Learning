import torch
import torch.nn as nn
import torch.optim as optim

# ===================== 直接用假数据，跳过 MNIST 下载 =====================
# 模拟 1000 张 28x28 的图片（和 MNIST 格式一样）
images = torch.randn(1000, 1, 28, 28)  # 1000张图     x
labels = torch.randint(0, 10, (1000,)) # 随机0-9标签   y

# ============================ 模型定义 ============================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============================ 初始化 ============================
model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================ 训练 ============================
model.train()
for epoch in range(5):
    # 前向传播
    outputs = model(images)
    loss = loss_fn(outputs, labels)

    # 反向更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"第{epoch+1}轮训练完成 | Loss: {loss.item():.4f}")

# ============================ 保存模型 ============================
torch.save(model.state_dict(), "day2_model.pth")
print("✅ 模型已保存：day2_model.pth")