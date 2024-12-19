import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd



df = pd.read_csv('e:/dataset/test2.0/test2.0/训练数据group3.csv')
# df = df.sample(n=100000, random_state=42)

original_labels = df['leakPipeId'].unique()
label_to_index = {label: idx for idx, label in enumerate(original_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
df['label_index'] = df['leakPipeId'].map(label_to_index)


df_x = df.iloc[:, :53]
df_x_normalized = (df_x - df_x.min()) / (df_x.max() - df_x.min())
df_x_normalized = df_x_normalized.fillna(0)
X = df_x_normalized.values
# 定义自编码器类
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Autoencoder, self).__init__()

        # 编码器：将输入映射到隐层（升维）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入到隐层
            nn.ReLU(),  # 激活函数
        )

        # 解码器：将隐层映射回输出空间（降维到输入维度）
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),  # 隐层到输出
            nn.Sigmoid()  # 激活函数，输出应与输入数据相同维度
        )

    def forward(self, x):
        # 通过编码器得到升维后的表示
        encoded = self.encoder(x)
        # 通过解码器重建输入
        decoded = self.decoder(encoded)
        return decoded, encoded


# 假设输入的维度为2，我们希望升维到4
input_dim = 53
hidden_dim = 128  # 设置升维的目标维度
output_dim = input_dim  # 解码器输出应该是原始维度
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实例化模型
model = Autoencoder(input_dim, hidden_dim, output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 将输入数据转换为Tensor并构建DataLoader
X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)
dataset = TensorDataset(X_train_tensor, X_train_tensor)  # 输入和标签都是X_train
batch_size = 128
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练自编码器
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        inputs, _ = data  # 获取输入数据（标签和输入数据相同）
        inputs = inputs.to(device)  # 将输入数据移到 GPU
        # 前向传播
        decoded, encoded = model(inputs)

        # 计算损失（重构误差）
        loss = criterion(decoded, inputs)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 保存模型的状态字典（只保存模型的权重）
torch.save(model.state_dict(), '../autoencoder_model.pth')
# 加载模型的状态字典
# model.load_state_dict(torch.load('autoencoder_model.pth'))
# 查看升维后的结果
with torch.no_grad():
    decoded, encoded = model(X_train_tensor)
    print("\nOriginal Input:\n", X_train_tensor)
    print("\nEncoded (升维后):\n", encoded)
    print("\nDecoded (重构后):\n", decoded)
