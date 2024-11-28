import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

# 数据预处理
file_path = "训练数据group3.csv"
data = pd.read_csv(file_path)

# 数据清洗
# 移除全为0的列
data_cleaned = data.loc[:, (data != 0).any(axis=0)]

# 填充缺失值，使用列均值填充
data_cleaned.fillna(data_cleaned.mean(), inplace=True)

# 映射标签
original_labels = data_cleaned['leakPipeId'].unique()
label_to_index = {label: idx for idx, label in enumerate(original_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
data_cleaned['label_index'] = data_cleaned['leakPipeId'].map(label_to_index)

# 转换特征和目标为 Tensor
X = data_cleaned.iloc[:, :-2].values
y = data_cleaned['label_index'].values

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA降维
pca = PCA(n_components=30)  # 设置合适的维度
X = pca.fit_transform(X)

# 转换为 Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 数据集划分为训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# ResNet模型定义
class ResNetClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_blocks=3, hidden_dim=128):
        super(ResNetClassifier, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # 定义多个残差块
        self.blocks = nn.ModuleList([self._make_resnet_block(hidden_dim) for _ in range(num_blocks)])
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def _make_resnet_block(self, dim):
        block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim, dim)
        )
        return block

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # 残差连接
        x = self.fc(x)
        return x

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X.shape[1]
num_classes = len(original_labels)
model = ResNetClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 早停机制
best_val_loss = float('inf')
patience = 3  # 容忍的 epoch 数
trigger_times = 0

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

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
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

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

# 保存模型
torch.save(model.state_dict(), 'resnet.pth')


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
