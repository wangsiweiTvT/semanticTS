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
from sklearn.metrics import mean_squared_error


# 定义超参数
df = pd.read_csv('/Users/wangsiwei/Desktop/训练数据group3.csv')
# df = df.sample(n=100000, random_state=42)


df_x = df.iloc[:, 0:53]
# df_x = np.log1p(df_x)
# df_x_normalized = (df_x - df_x.min()) / (df_x.max() - df_x.min())
df_x_normalized = (df_x - df_x.mean()) / df_x.std()
df_x_normalized = df_x_normalized.fillna(0)

y = df['leakage']/10
y_normalized = (y - y.mean()) / y.std()
y_normalized = y_normalized.fillna(0)
# 创建RandomProjection对象，设置目标维度为128
# transformer = SparseRandomProjection(n_components=1500)
# 对输入向量进行升维
# X = transformer.fit_transform(df_x_normalized)
# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(df_x_normalized)

# 特征和目标变量
# X = df_x_normalized.values
y = scaler.fit_transform([[yi] for yi in y_normalized])
# PCA降维
# pca = PCA(n_components=1500)  # 设置合适的维度
# X = pca.fit_transform(X)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 转换为 Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048)
test_loader = DataLoader(test_dataset, batch_size=2048)


# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size1, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc4(out)
        return out



# 初始化模型、损失函数和优化器
input_size = X.shape[1]
hidden_size1 = 512
# hidden_size2 = 512
# hidden_size3 = 2048
model = NeuralNet(input_size, hidden_size1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# 早停机制
best_val_loss = float('inf')
patience = 5  # 容忍的 epoch 数
trigger_times = 0
# 训练模型
num_epochs = 500
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
    scheduler.step(val_loss)

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
torch.save(model, 'mlp_reg_model.pth')

# 测试模型
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    all_predictions = []
    all_targets = []
    for X_batch, Y_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        predictions = model(X_batch)
        all_predictions.append(predictions.cpu())
        all_targets.append(Y_batch.cpu())

        # 将所有批次的结果拼接在一起
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mse = mean_squared_error(all_targets, all_predictions)
    rmse = mse ** 0.5

    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')

