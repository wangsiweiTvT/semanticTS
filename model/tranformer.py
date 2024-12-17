import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import pandas as pd
from util.BPEutil import discretize_series,create_value_to_unicode_map,convert_to_unicode


# 检查是否有可用的 GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 示例数据
texts = []

# ETT
df = pd.read_csv('/Users/wangsiwei/Desktop/sematics4TS/ETT-small/ETTm1.csv')
time_series1 = df['HUFL']
time_series2 = df['HULL']
time_series3 = df['MUFL']
time_series4 = df['MULL']
time_series5 = df['LUFL']
time_series6 = df['LULL']
time_series7 = df['OT']

time_series = time_series5
print(time_series.size)
timestamp = list(range(0, time_series.size, 1))


#加载 BPE 分词器
tokenizer = Tokenizer.from_file("../util/bpe-tokenizer.json")

num_bins = 120
# 将时间序列数据离散化为索引 [478 478 501 503 485 465 440 431 395 366 333 329 348 387 368 275 229 222]
symbols = discretize_series(time_series, num_bins)
# print(f"Discretized symbols: {symbols}")

value2unicode, unicode2value = create_value_to_unicode_map(symbols)
# 将符号序列转换为字符串形式
unicode_series = convert_to_unicode(symbols, value2unicode)
# print(f"unicode_series: {unicode_series}")
symbol_str = ''.join(unicode_series)

# 编码
encoded = tokenizer.encode(symbol_str)

# vocab = tokenizer.get_vocab()


# 解码
decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
print(decoded)
tokens = decoded.split()
print(len(tokens))

for i in range(2,len(tokens)):
    for j in range(0,len(tokens)-i):
        texts.append(''.join(tokens[j:j+i]))

print(len(texts))


# 使用分词器对文本进行编码
def encode_texts(texts, tokenizer):
    return [tokenizer.encode(text).ids for text in texts]

encoded_texts = encode_texts(texts, tokenizer)

# 数据集定义
class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx])

# 模型定义
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output)

# 参数设置
vocab_size = tokenizer.get_vocab_size()  # 词汇表大小
model_dim = 512   # 模型维度
num_heads = 4     # 注意力头数
num_layers = 4    # 编码器和解码器层数
output_dim = vocab_size # 输出词汇表大小

# 数据
data = encoded_texts[:-1]  # 示例数据
targets = encoded_texts[1:]  # 示例目标数据

dataset = SimpleDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 初始化模型、损失函数和优化器
model = TransformerModel(vocab_size, model_dim, num_heads, num_layers, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:].reshape(-1)

        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.view(-1, output_dim), tgt_output)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 使用模型进行预测
src = torch.tensor([encoded_texts[0]]).to(device)  # 示例输入
tgt_input = torch.tensor([encoded_texts[1][:-1]]).to(device)  # 示例目标输入
output = model(src, tgt_input)
predicted = torch.argmax(output, dim=-1)

# 解码预测结果
decoded_prediction = tokenizer.decode(predicted[0].tolist())
print("Predicted:", decoded_prediction)
