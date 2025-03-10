import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# 假设每个文件夹下的文件是时间序列数据（.csv  格式），
# 每个文件夹代表一个类别
class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # 遍历每个类别文件夹
        self.label_encoder = LabelEncoder()
        self.class_names = os.listdir(data_dir)
        self.class_names.sort()  # 确保标签排序一致

        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)

            # 遍历类别文件夹下的每个文件
            for sample_dir_name in os.listdir(class_dir):
                sample_dir_path = os.path.join(class_dir, sample_dir_name)
                csvs = os.listdir(sample_dir_path)
                for i in csvs :
                    if i[0:9] == 'data_left':
                        file_path = os.path.join(sample_dir_path, i)
                # 读取文件内容，假设为 .csv 文件
                # 你可以根据数据格式调整读取方式（如 .npy, .csv 等）
                if file_path.endswith(".csv"):
                    data =pd.read_csv(file_path)['CH17'].values

                # 将数据添加到列表

                self.data.append(data)
                self.labels.append(label)  # 标签是该类文件夹的索引

        # 转换为 NumPy 数组并归一化或标准化（可选）
        self.data = np.expand_dims(self.data,-1)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        # 进行训练集和测试集的划分
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42
        )

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        data = torch.tensor(self.X_train[idx], dtype=torch.float32)
        label = torch.tensor(self.y_train[idx], dtype=torch.long)

        if self.transform:
            data = self.transform(data)

        return data, label


# 示例：读取训练数据
data_dir = "E:/dataset/BJTU_RAO_Bogie_Datasets"  # 你的数据路径
dataset = TimeSeriesDataset(data_dir)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

#
# class TCN(nn.Module):
#     def __init__(self, input_size, num_classes, num_channels=[64, 128, 256], kernel_size=2, dilation_rates=[1, 2, 4]):
#         super(TCN, self).__init__()
#
#         # 定义 TCN 的层（1D 卷积层）
#         layers = []
#         in_channels = input_size
#         for i, out_channels in enumerate(num_channels):
#             dilation = dilation_rates[i]  # 获取当前层的膨胀因子
#
#             # 因果卷积：在每一层的卷积中应用膨胀卷积，并设置padding为dilation的大小
#             layers.append(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=dilation,
#                           dilation=dilation)
#             )
#             layers.append(nn.ReLU())
#             layers.append(nn.MaxPool1d(kernel_size=2, stride=2))  # 可选：池化层
#             in_channels = out_channels
#
#         self.tcn_layers = nn.Sequential(*layers)
#
#         # 分类头部
#         self.fc = nn.Linear(num_channels[-1], num_classes)
#
#     def forward(self, x):
#         # 输入的 x 是 (batch_size, seq_len, input_size)，需要调整为 (batch_size, input_size, seq_len)
#         x = x.permute(0, 2, 1)  # 改变为 (batch_size, input_size, seq_len)
#
#         x = self.tcn_layers(x)
#
#         # 全局池化
#         x = x.mean(dim=2)  # (batch_size, num_channels[-1])
#
#         # 分类
#         x = self.fc(x)
#
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """TCN 残差块（膨胀因果卷积 + 归一化 + 激活 + Dropout）"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # 因果卷积所需的左侧填充

        # 主要卷积路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        # 下采样连接（当输入输出通道数不一致时）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """权重初始化"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x
        # 第一个卷积层
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # 裁剪右侧多余的填充
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        # 第二个卷积层
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNClassifier(nn.Module):
    """TCN 分类网络"""

    def __init__(self, input_size, num_classes, num_channels = [64,64,64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        # 堆叠多个 TemporalBlock
        for i in range(num_levels):
            dilation_size = 2 ** i  # 指数增长的膨胀系数
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     dilation=dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        # 全局平均池化 + 全连接层
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # 输入的 x 是 (batch_size, seq_len, input_size)，需要调整为 (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)  # 改变为 (batch_size, input_size, seq_len)
        # 输入形状: (batch_size, input_size, seq_len)
        x = self.tcn(x)  # 输出形状: (batch_size, num_channels[-1], seq_len)
        x = self.gap(x).squeeze()  # 全局平均池化: (batch_size, num_channels[-1])
        return self.fc(x)  # 分类输出: (batch_size, num_classes)

import torch.optim as optim
from sklearn.metrics import accuracy_score

# 检查是否有 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型
input_size = dataset.X_train.shape[2]  # 每个时间序列的特征数
# input_size = 1  # 每个时间序列的特征数
num_classes = len(dataset.class_names)
model = TCNClassifier(input_size=input_size, num_classes=num_classes)

# 将模型移到 GPU（如果有的话）
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算准确度
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 计算每个 epoch 的准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

torch.save(model, 'tcn_single_class.pth')
# 测试代码
model.eval()
test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 计算准确度
test_acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {test_acc:.2f}%")
