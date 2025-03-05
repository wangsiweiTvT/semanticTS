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

predict_len = 100

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

time_series = time_series6


time_series = time_series.fillna(0)

# time_series = np.random.rand(100000)
# 合成 多周期性 序列
# timestamp, ts, s, t = synthesis(100000, 0.000, [25,30, 40, 125, 150,500], [1,1, 1, 1, 1,1],0.0)
# time_series=ts
plt.plot(range(0, len(time_series)), time_series, color='blue')
plt.show()
num_bins = 10000
# 去高频
# time_series = freq_filter(time_series, 6000)
bins = np.linspace(min(time_series), max(time_series), num_bins )
# 将时间序列数据离散化为索引 [478 478 501 503 485 465 440 431 395 366 333 329 348 387 368 275 229 222]
symbols = discretize_series(time_series, num_bins)
print(len(symbols))
timestamp = list(range(0, len(symbols)))

timestamp1, s1, s, t = synthesis(100, 0.000, [7,17], [1,1],0.0)
timestamp11, s1t1, s, t = synthesis(100, 0.1, [7,17], [1,1],0.0)

s1symbols = discretize_series(s1, num_bins)
s1t1symbols = discretize_series(s1t1, num_bins)
input_s1 = torch.tensor([s1symbols ]).to(device)
input_s1t1 = torch.tensor([s1t1symbols ]).to(device)

timestamp2, s2, s, t = synthesis(100, 0.000, [5,23], [1,1],0.0)
timestamp22, s2t2, s, t = synthesis(100, 0.1, [5,23], [1,1],0.0)

s2symbols = discretize_series(s2, num_bins)
s2t2symbols = discretize_series(s2t2, num_bins)
input_s2 = torch.tensor([s2symbols ]).to(device)
input_s2t2 = torch.tensor([s2t2symbols ]).to(device)


plt.subplot(2, 2, 1)
plt.plot(range(0, len(s1)), s1)

plt.subplot(2, 2, 3)
plt.plot(range(0, len(s1t1)), s1t1)

plt.subplot(2, 2, 2)
plt.plot(range(0, len(s2)), s2)

plt.subplot(2, 2, 4)
plt.plot(range(0, len(s2t2)), s2t2)


plt.show()
# encoder inpout [0:100] decoder input [99:199] decoder output[100:200]
for i in range(predict_len*2,predict_len*2+1):
    for j in range(0,len(symbols)-i,10):
        texts.append(symbols[j:j+i])

# for j in range(0,len(tokens)-10):
#     texts.append(''.join(tokens[j:j+10]))

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
        input_seq = torch.tensor(text[:-predict_len])
        target_seq = torch.tensor(text[-predict_len:])
        return input_seq, target_seq


def collate_fn(batch):
    inputs, targets = zip(*batch)

    # 找到当前批次中最长的序列
    max_len_in = max(len(seq) for seq in inputs)
    max_len_out = max(len(seq) for seq in targets)

    # 填充序列
    padded_inputs = [torch.cat([seq, torch.zeros(max_len_in - len(seq), dtype=torch.long)]) for seq in inputs]
    padded_targets = [torch.cat([seq, torch.zeros(max_len_out - len(seq), dtype=torch.long)]) for seq in targets]

    return torch.stack(padded_inputs), torch.stack(padded_targets)


dataset = TextDataset(train_texts)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

dataset_test = TextDataset(test_texts)
data_loader_test = DataLoader(dataset_test, batch_size=64, shuffle=True, collate_fn=collate_fn)


# 模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_size, num_heads, hidden_size, num_layers):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, hidden_size,batch_first=True)
        self.fc = nn.Linear(embed_size, input_size)
        # 权重共享
        # self.fc.weight = self.embedding.weight
    def forward(self, src, tgt):
        # src = self.embedding(src) * math.sqrt(self.embed_size)
        # src = self.pos_encoder(src)

        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)
        # src_key_padding_mask = TransformerModel.get_key_padding_mask(src).to(device)
        # tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt).to(device)

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
            # 将真值targets作为解码器输入 是 teach force过程  在训练过程 不管是多步还是单步 都是采用 teach force 过程
            outputs = model(inputs,targets[:,:-1])
            loss = criterion(outputs.view(-1, num_bins), targets[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')

    torch.save(model, 'ts-point-transformer-nopos-stride100.pth')

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
            decoder_input = torch.ones(targets[:,0].unsqueeze(1).size()).long().to(device) #(64,100)
            decoder_input[:, 0] = targets[:,0]
            predictions = []
            predictions.append(decoder_input[:, 0])
            for t in range(predict_len-1):
                output = model(inputs, decoder_input) # (64,100,10000)
                predicted = output.argmax(dim=-1)#(64,100)
                # decoder_input[:,t+1] = predicted[:,-1]  # 使用当前时间步的预测值作为下一个时间步的输入
                decoder_input = torch.cat((decoder_input, predicted[:,-1].unsqueeze(1)), dim=1)
                predictions.append(predicted[:,-1])


            predictions = torch.stack(predictions).t()
            # outputs = model(inputs, targets)
            # loss = criterion(outputs.view(-1, num_bins), targets.view(-1))
            # total_loss += loss.item()

            # predicted = outputs.argmax(dim=-1)
            # correct += (predicted == targets).sum().item()
            total += targets.numel()

            if(total%2==0):
                series_in = inputs[0].cpu().numpy()
                series_out = predictions[0].cpu().numpy()
                series_truth = targets[0].cpu().numpy()
                plt.plot(range(0,len(series_in)), series_in,color = 'blue')
                plt.plot(range(len(series_in),len(series_in)+len(series_out)), series_out,color = 'red')
                plt.plot(range(len(series_in),len(series_in)+len(series_truth)), series_truth,color = 'green')
                plt.show()

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 假设 input_size, embed_size, num_heads, hidden_size, num_layers 等参数已定义
    input_size = num_bins
    embed_size = 512
    num_heads = 4
    hidden_size = 2048
    num_layers = 4

    model = TransformerModel(input_size, embed_size, num_heads, hidden_size, num_layers).to(device)
    model = torch.load('ts-point-transformer-nopos-stride100.pth', map_location=torch.device(device))


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 200
    # train(model, data_loader, criterion, optimizer, num_epochs, device)

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


