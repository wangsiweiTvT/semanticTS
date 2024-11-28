import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report



df = pd.read_csv('/Users/wangsiwei/Desktop/训练数据group3.csv')
# df = df.sample(n=100000, random_state=42)

original_labels = df['leakPipeId'].unique()
label_to_index = {label: idx for idx, label in enumerate(original_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
df['label_index'] = df['leakPipeId'].map(label_to_index)


df_x = df.iloc[:, :53]
df_x_normalized = (df_x - df_x.min()) / (df_x.max() - df_x.min())
df_x_normalized = df_x_normalized.fillna(0)
# print(df_x_normalized.describe())
# pca = PCA(n_components=30)  # 设置合适的维度
# X = pca.fit_transform(df_x)
# 特征和目标变量
# X = df_x_normalized.iloc[:, :35]
# X = df_x_normalized.iloc[:, 35:53]
X = df_x_normalized
y_class = df['label_index']

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转换为 Tensor
X = torch.tensor(X, dtype=torch.float32) # 添加一个维度以匹配 Transformer 输入格式
y = torch.tensor(y_class, dtype=torch.long)

y_reg = df['leakage']
X_train, X_temp, y_train, y_temp = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 标准化特征
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 转换为张量
# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#
#
#
# X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

# 创建数据加载器
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)



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
hidden_size = 256
num_classes = len(original_labels)
model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# 早停机制
best_val_loss = float('inf')
patience = 3  # 容忍的 epoch 数
trigger_times = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:

        inputs, targets = X_batch.to(device), y_batch.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #
        # # 前向传播
        # outputs = model(X_batch)
        # loss = criterion(outputs, y_batch)
        #
        # # 反向传播和优化
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    # if (epoch + 1) % 10 == 0:
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # 学习率调度器更新
    scheduler.step()

    # 早停逻辑
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        print(f"Early stopping patience count: {trigger_times}")
        if trigger_times >= patience:
            print("Early stopping triggered. Stopping training.")
            break

# 假设 model 是你的神经网络模型
torch.save(model, 'dnn_pca_model.pth')

# 测试模型
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# 映射回原始标签
all_preds_original = [index_to_label[pred] for pred in all_preds]
all_targets_original = [index_to_label[target] for target in all_targets]

# 分类报告
print("\n=== 分类报告 ===")
print(classification_report(all_targets_original, all_preds_original))

# # 测试模型
# with torch.no_grad():
#     outputs = model(X_test_tensor)
#     _, predicted = torch.max(outputs, 1)
#     accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
#     print(f'测试集准确度: {accuracy:.2f}')
