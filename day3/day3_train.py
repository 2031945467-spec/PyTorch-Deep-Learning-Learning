import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.randn(1000, 1, 28, 28)
y_train = torch.randint(0, 10, (1000,))
x_test = torch.randn(200, 1, 28, 28)
y_test = torch.randint(0, 10, (200,))


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
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(10):
    pred=model(x_train)
    loss=loss_fn(pred,y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"Epoch {epoch + 1} | Loss: {loss.item():.3f}")

model.eval()
with torch.no_grad():
    outputs=model(x_test)
    predictions=torch.argmax(outputs,dim=1)
    correct=(predictions==y_test).sum().item()
    acc=correct/len(x_test)
print(f"\n测试集准确率：{acc*100:.2f}%")
torch.save(model.state_dict(), "day3_model.pth")
print("模型已保存：day3_model.pth")