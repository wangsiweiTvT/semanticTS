import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np
import pandas as pd
from sklearn.random_projection import SparseRandomProjection

# 注意力机制模块 (Self-Attention)
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, 1))  # 注意力权重参数

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_weights = torch.softmax(attention_scores, dim=0)
        weighted_sum = torch.sum(attention_weights * x, dim=0)
        return weighted_sum

# 改进后的神经网络模型
class ImprovedNeuralNetWithLayers(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ImprovedNeuralNetWithLayers, self).__init__()
        
        # 第一层线性变换 + Batch Normalization + Swish激活函数
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.swish1 = nn.SiLU()  # 使用Swish激活函数

        # 残差连接（Residual Connection） 1,7
        self.res_fc1 = nn.Linear(input_size, num_classes)  # 用于残差连接的线性层

        # 第二层线性变换 + Batch Normalization + Swish激活函数
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.swish2 = nn.SiLU()  # 使用Swish激活函数

        # 残差连接（Residual Connection）2,6
        self.res_fc2 = nn.Linear(hidden_size, hidden_size)  # 用于残差连接的线性层

        # 第三层线性变换 + Batch Normalization + Swish激活函数
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*4)
        self.bn3 = nn.BatchNorm1d(hidden_size*4)
        self.swish3 = nn.SiLU()  # 使用Swish激活函数

        # 残差连接（Residual Connection） 3,5
        self.res_fc3 = nn.Linear(hidden_size*2, hidden_size*2)  # 用于残差连接的线性层

        # 第四层线性变换 + Batch Normalization + Swish激活函数
        self.fc4 = nn.Linear(hidden_size*4, hidden_size*8)
        self.bn4 = nn.BatchNorm1d(hidden_size*8)
        self.swish4 = nn.SiLU()  # 使用Swish激活函数

        # 残差连接（Residual Connection）
        self.res_fc4 = nn.Linear(hidden_size*4, hidden_size*4)  # 用于残差连接的线性层


        # 第五层线性变换 + Batch Normalization + Swish激活函数
        self.fc5 = nn.Linear(hidden_size*8, hidden_size*4)
        self.bn5 = nn.BatchNorm1d(hidden_size*4)
        self.swish5 = nn.SiLU()  # 使用Swish激活函数

        # 残差连接（Residual Connection）
        # self.res_fc5 = nn.Linear(hidden_size*8, hidden_size*4)  # 用于残差连接的线性层


        # 自注意力机制
        self.attention = SelfAttention(hidden_size*4)

        # 第六层线性变换 + Batch Normalization + Swish激活函数
        self.fc6 = nn.Linear(hidden_size*4, hidden_size*2)
        self.bn6 = nn.BatchNorm1d(hidden_size*2)
        self.swish6 = nn.SiLU()  # 使用Swish激活函数

        # 残差连接（Residual Connection）
        # self.res_fc6 = nn.Linear(hidden_size*4, hidden_size*2)  # 用于残差连接的线性层

        # 第七层线性变换 + Batch Normalization + Swish激活函数
        self.fc7 = nn.Linear(hidden_size*2, hidden_size)
        self.bn7 = nn.BatchNorm1d(hidden_size)
        self.swish7 = nn.SiLU()  # 使用Swish激活函数

        # 残差连接（Residual Connection）
        # self.res_fc7 = nn.Linear(hidden_size*2, hidden_size)  # 用于残差连接的线性层


        # 输出层
        self.fc8 = nn.Linear(hidden_size, num_classes)
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(0.5)

        # 权重初始化
        self._initialize_weights()

    def forward(self, x):
        # 第一部分
        out1 = self.fc1(x)
        out1 = self.bn1(out1)
        out1 = self.swish1(out1)
        
        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        out2 = self.swish2(out2)

        out3 = self.fc3(out2)
        out3 = self.bn3(out3)
        out3 = self.swish3(out3)

        # 后续层
        out4 = self.fc4(out3)
        out4 = self.bn4(out4)
        out4 = self.swish4(out4)

        # 自注意力机制 （备选，loss干到20多的罪魁祸首可能是它）
        #attention_out = self.attention(out4)  # 引入注意力机制
        #out4 = out4 + attention_out  # 将注意力加回输出

        out5 = self.fc5(out4)
        out5 = self.bn5(out5)
        out5 = self.swish5(out5)
        #5+3
        res_out5 = self.res_fc4(out3)
        out5 = out5 + res_out5
        
        out6 = self.fc6(out5)
        out6 = self.bn6(out6)
        out6 = self.swish6(out6)
        #6+2
        res_out6 = self.res_fc3(out2)
        out6 = out6 + res_out6

        out7 = self.fc7(out6)
        out7 = self.bn7(out7)
        out7 = self.swish7(out7)
        #7+1
        res_out7 = self.res_fc2(out1)
        out7 = out7 + res_out7

        out7 = self.dropout(out7)
        
        out8 = self.fc8(out7)
        return out8

    def _initialize_weights(self):
        # 使用 Xavier 初始化方法进行权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

if __name__ == "__main__":
    # 加载数据
    file_path = "e:/dataset/test2.0/test2.0/训练数据group3.csv"
    data = pd.read_csv(file_path)

    # 数据清洗
    data_cleaned = data
    original_labels = data_cleaned['leakPipeId'].unique()
    label_to_index = {label: idx for idx, label in enumerate(original_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    data_cleaned['label_index'] = data_cleaned['leakPipeId'].map(label_to_index)

    # 特征与目标分离
    X = data_cleaned.iloc[:, :-3].values
    y = data_cleaned['label_index'].values

    # 创建RandomProjection对象，设置目标维度为
    transformer = SparseRandomProjection(n_components=1024)
    # 对输入向量进行升维
    X = transformer.fit_transform(X)
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # PCA降维
    pca = PCA(n_components=1024)
    X = pca.fit_transform(X)

    # 转换为Tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    # 模型初始化并放入GPU中
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    num_classes = len(original_labels)
    model = ImprovedNeuralNetWithLayers(input_dim, hidden_size=512, num_classes=num_classes).to(device)

    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  # AdamW 比 Adam 更加稳定

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 停止机制，超10次停止
    best_val_loss = float('inf')
    patience = 10 
    trigger_times = 0

    # 训练模型
    num_epochs = 2000
    for epoch in range(num_epochs):
        # 训练
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

        # 验证
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

        # 更新学习率
        scheduler.step(val_loss)

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
    torch.save(model.state_dict(), 'improved_model_with_layers.pth')

    # 保存Scaler和PCA
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')

    # 保存Label映射字典
    joblib.dump(label_to_index, 'label_to_index.pkl')
    joblib.dump(index_to_label, 'index_to_label.pkl')

    # 测试
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
