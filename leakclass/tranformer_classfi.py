import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
# 数据预处理
file_path = "/Users/wangsiwei/Desktop/训练数据group3.csv"
data = pd.read_csv(file_path)

# 映射标签
original_labels = data['leakPipeId'].unique()
label_to_index = {label: idx for idx, label in enumerate(original_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
data['label_index'] = data['leakPipeId'].map(label_to_index)

# 转换特征和目标为 Tensor
X = data.iloc[:, :-2].values
y = data['label_index'].values

# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转换为 Tensor
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # 添加一个维度以匹配 Transformer 输入格式
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

# Transformer 模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=4, hidden_dim=256, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)  # 映射到隐藏维度
        x = self.transformer(x)  # Transformer 编码器
        x = x.mean(dim=1)  # 平均池化
        return self.classifier(x)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X.shape[2]
num_classes = len(original_labels)
model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
torch.save(model, '../transformer_model.pth')
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
