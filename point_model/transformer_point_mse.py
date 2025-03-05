import torch
import torch.nn as nn
import torch.optim as optim
from scipy import spatial
import math
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import pandas as pd
from util.BPEutil import discretize_series,create_value_to_unicode_map,convert_to_unicode,restore_tokens,restore_series,convert_to_discretized
import random
import numpy as np
from util.synthesisTS import freq_filter,synthesis
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter



# 检查是否有可用的 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 示例数据
texts = []

# ETT
df = pd.read_csv('E:/ideaproj/ETDataset/ETT-small/ETTm1.csv')
time_series1 = df['HUFL']
time_series2 = df['HULL']
time_series3 = df['MUFL']
time_series4 = df['MULL']
time_series5 = df['LUFL']
time_series6 = df['LULL']
time_series7 = df['OT']

# time_series4  time_series3 time_series2 epoch=10  预测的特别好。 其他的一般 为什么？ 最大最小差值很大

#time_series1 epoch=20 就变得好了

# time_series7 最大最小差值40  Epoch [1/10], Loss: 123.7240   第一轮loss 很大    epoch=10 时 结果不好  但比5,6好  epoch=50 变好了
# time_series6 最大最小差值4   Epoch [1/10], Loss: 5.4304     第一轮loss 比较小  epoch=20 时 结果不好，  epoch=50 变好了
# time_series5 最大最小差值8   Epoch [1/10], Loss: 5.4043     第一轮loss 比较小  epoch=10 时 结果不好，  epoch=50 变好了

# time_series4 最大最小差值14  Epoch [1/10], Loss: 3.3138     第一轮loss 比较小  epoch=10 时 结果比567好，
# time_series3 最大最小差值40  Epoch [1/10], Loss: 36.9944    第一轮loss 比较大  epoch=10 时 结果比567好，
# time_series2 最大最小差值16  Epoch [1/10], Loss: 5.6590     第一轮loss 比较小  epoch=10 时 结果比567好，
# time_series1 最大最小差值40  Epoch [1/10], Loss: 45.7887    第一轮loss 比较大  epoch=10 时 结果比567好


time_series = time_series7


time_series = time_series.fillna(0)

plt.plot(range(0,len(time_series)),time_series)
plt.show()

for i in range(200,201):
    for j in range(0,len(time_series)-i,100):
        texts.append(time_series[j:j+i].to_numpy())

random.seed(42)
def split_list(data, train_ratio):
    random.shuffle(data)
    train_size = int(train_ratio * len(data))
    train_list = data[:train_size]
    test_list = data[train_size:]
    return train_list, test_list

train_texts, test_texts = split_list(texts, train_ratio=0.5)
print(len(train_texts))



# 数据集定义
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_seq = torch.tensor(text[:-100],dtype=torch.float32)
        target_seq = torch.tensor(text[-100:],dtype=torch.float32)
        return input_seq, target_seq


def collate_fn(batch):
    inputs, targets = zip(*batch)
    # 找到当前批次中最长的序列
    max_len_in = max(len(seq) for seq in inputs)
    max_len_out = max(len(seq) for seq in targets)

    # 填充序列
    padded_inputs = [torch.cat([seq, torch.zeros(max_len_in - len(seq))]) for seq in inputs]
    padded_targets = [torch.cat([seq, torch.zeros(max_len_out - len(seq))]) for seq in targets]


    input = torch.stack(padded_inputs).reshape(len(inputs), max_len_in, 1)

    output = torch.stack(padded_targets).reshape(len(targets), max_len_out, 1)

    # return torch.stack(padded_inputs), torch.stack(padded_targets)
    return input, output


dataset = TextDataset(train_texts)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

dataset_test = TextDataset(test_texts)
data_loader_test = DataLoader(dataset_test, batch_size=64, shuffle=True, collate_fn=collate_fn)


# 模型定义
class TransformerModel(nn.Module):
    def __init__(self,  output_size,embed_size, num_heads, hidden_size, num_layers):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.output_size = output_size
        self.embedding = nn.Linear(1, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, hidden_size,batch_first=True)
        self.fc = nn.Linear(embed_size, 1)

    def forward(self, src, tgt):
        # src = self.embedding(src) * math.sqrt(self.embed_size)
        # src = self.pos_encoder(src)

        # src_key_padding_mask = TransformerModel.get_key_padding_mask(src).to(device)
        # tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt).to(device)
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(device)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # 将准备好的数据送给transformer
        output = self.transformer(src, tgt,
                                  tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

    def encode(self,src):
        src_key_padding_mask = TransformerModel.get_key_padding_mask(src).to(device)
        src = self.embedding(src)
        return self.transformer.encoder.forward(src,src_key_padding_mask=src_key_padding_mask)

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 0] = -torch.inf
        return key_padding_mask

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


def train(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs,targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')

    torch.save(model, 'ts-point-transformer-nopos-mse-stride100.pth')

# def create_mask(src, pad_token=0):
#     src_seq_len = src.shape[1]
#     src_mask = (src != pad_token).unsqueeze(1).repeat(1, src_seq_len, 1)
#     return src_mask


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)# inputs (64,100)
            outputs = model(inputs, targets)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            if(total%2==0):
                series_in = inputs[0].cpu().numpy()
                series_out = outputs[0].cpu().numpy()
                series_truth = targets[0].cpu().numpy()
                plt.plot(range(0,len(series_in)), series_in,color = 'blue')
                plt.plot(range(len(series_in),len(series_in)+len(series_out)), series_out,color = 'red')
                plt.plot(range(len(series_in),len(series_in)+len(series_truth)), series_truth,color = 'green')
                plt.show()

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 假设 input_size, embed_size, num_heads, hidden_size, num_layers 等参数已定义
    input_size = 1
    seq_len = 100
    output_size = 100
    embed_size = 512
    num_heads = 4
    hidden_size = 2048
    num_layers = 4

    model = TransformerModel(output_size, embed_size, num_heads, hidden_size, num_layers).to(device)
    # model = torch.load('ts-point-transformer-stride100.pth', map_location=torch.device(device))


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 20
    train(model, data_loader, criterion, optimizer, num_epochs, device)

    # with torch.no_grad():
    #     v1 = model.encode(input_s1).cpu().numpy()
    #     v11 = model.encode(input_s1t1).cpu().numpy()
    #     v2 = model.encode(input_s2).cpu().numpy()
    #     v22 = model.encode(input_s2t2).cpu().numpy()
    #     print(v1[0][99])
    #     print(v11[0][99])
    #     print(v2[0][99])
    #     print(v22[0][99])
    #     minus_similar = 1 - spatial.distance.cosine(v11[0][99]-v1[0][99], v22[0][99]-v2[0][99])
    #     print('v11-v1', ',', 'v22-v2', minus_similar)


    evaluate(model, data_loader_test, criterion, device)


if __name__ == "__main__":
    main()


