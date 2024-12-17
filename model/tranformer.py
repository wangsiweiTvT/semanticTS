import torch
import torch.nn as nn
import torch.optim as optim
import math
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

for i in range(3,10):
    for j in range(0,len(tokens)-i):
        texts.append(''.join(tokens[j:j+i]))

print(len(texts))


# 使用分词器对文本进行编码
def encode_texts(texts, tokenizer):
    return [tokenizer.encode(text).ids for text in texts]

encoded_texts = encode_texts(texts, tokenizer)



# 数据集定义
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_seq = torch.tensor(text[:-1])
        target_seq = torch.tensor(text[1:])
        return input_seq, target_seq


def collate_fn(batch):
    inputs, targets = zip(*batch)

    # 找到当前批次中最长的序列
    max_len = max(len(seq) for seq in inputs)

    # 填充序列
    padded_inputs = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in inputs]
    padded_targets = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in targets]

    return torch.stack(padded_inputs), torch.stack(padded_targets)


dataset = TextDataset(encoded_texts)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


# 模型定义
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, hidden_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        # src = self.embedding(src) * math.sqrt(self.embed_size)
        # src = self.pos_encoder(src)

        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[0])
        src_key_padding_mask = TransformerModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # 将准备好的数据送给transformer
        output = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc(output)
        return output

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 0] = -torch.inf
        return torch.transpose(key_padding_mask,0,1)

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
            loss = criterion(outputs.view(-1, tokenizer.get_vocab_size()), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')


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
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, targets)
            loss = criterion(outputs.view(-1, tokenizer.get_vocab_size()), targets.view(-1))
            total_loss += loss.item()

            predicted = outputs.argmax(dim=-1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

    accuracy = correct / total
    print(f'Loss: {total_loss / len(data_loader):.4f}, Accuracy: {accuracy:.4f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 假设 vocab_size, embed_size, num_heads, hidden_size, num_layers 等参数已定义
    vocab_size = tokenizer.get_vocab_size()
    embed_size = 512
    num_heads = 4
    hidden_size = 2048
    num_layers = 4

    model = TransformerModel(vocab_size, embed_size, num_heads, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train(model, data_loader, criterion, optimizer, num_epochs, device)
    evaluate(model, data_loader, criterion, device)


if __name__ == "__main__":
    main()


