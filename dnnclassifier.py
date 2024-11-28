import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/wangsiwei/Desktop/训练数据group3.csv')
# df = df.sample(n=100000, random_state=42)

df_x = df.iloc[:, :53]
df_x_normalized = (df_x - df_x.min()) / (df_x.max() - df_x.min())
df_x_normalized = df_x_normalized.fillna(0)
# print(df_x_normalized.describe())

# 特征和目标变量
# X = df_x_normalized.iloc[:, :35]
# X = df_x_normalized.iloc[:, 35:53]
X = df_x_normalized
y_class = df['leakPipeId']
y_reg = df['leakage']
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 初始化模型、损失函数和优化器
input_size = X_train_scaled.shape[1]
hidden_size = 64
num_classes = 314
model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 假设 model 是你的神经网络模型
torch.save(model, 'dnn_model.pth')
# 测试模型
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'测试集准确度: {accuracy:.2f}')
