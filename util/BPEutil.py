import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import pandas as pd
import matplotlib.pyplot as plt
from synthesisTS import synthesis
import unicodedata

num =0
def create_value_to_unicode_map(discretized_data):
    unique_values = sorted(set(value for value in discretized_data))
    value2unicode = {value: chr(i + 0x10000) for i, value in enumerate(unique_values)}
    unicode2value = {chr(i + 0x10000): value for i, value in enumerate(unique_values)}
    return value2unicode,unicode2value

def convert_to_unicode(discretized_data, value2unicode):
    unicode_series = [value2unicode[value] for value in discretized_data]
    return unicode_series


def convert_to_discretized(decoded, unicode2value):
    idx = list()
    num = 0
    for value in decoded:
        if value==' ':continue
        idx.append(unicode2value[value])

    return idx
# 离散化（将数值范围分成若干区间）
def discretize_series(series, num_bins):
    bins = np.linspace(min(series), max(series), num_bins + 1)
    symbols = np.digitize(series, bins) - 1  # 转换为符号索引
    return symbols

# 还原时间序列数据
def restore_series(symbols, bins):
    restored_series = [np.mean(bins[s:s+2]) for s in symbols]
    return np.array(restored_series)

if __name__ == '__main__':
    df = pd.read_csv('../2018aiops_dataset/8a20c229e9860d0c_35136.csv')
    time_series = df.head(800)['value']
    # timestamp, time_series, s, t = synthesis(1000, 0.000, [50, 100, 200, 400], [1, 1, 2, 2])
    timestamp = list(range(0, 800, 1))
    plt.subplot(2, 2, 1)

    plt.plot(timestamp, time_series)

    num_bins = len(time_series)
    # 将时间序列数据离散化为索引 [478 478 501 503 485 465 440 431 395 366 333 329 348 387 368 275 229 222]
    symbols = discretize_series(time_series, num_bins)
    print(f"Discretized symbols: {symbols}")

    value2unicode,unicode2value = create_value_to_unicode_map(symbols)
    print("Value to Unicode Mapping:", value2unicode)
    print("Unicode to Value Mapping:", unicode2value)
    # 将符号序列转换为字符串形式
    unicode_series = convert_to_unicode(symbols, value2unicode)
    print(f"unicode_series: {unicode_series}")
    symbol_str = ''.join(unicode_series)

    print(f"symbol_str: {symbol_str}")
    print(len(symbol_str))
    # 初始化BPE模型
    BPE_model = models.BPE()
    tokenizer = Tokenizer(BPE_model)

    # 定义预处理器和训练器
    trainer = trainers.BpeTrainer(vocab_size=num_bins, min_frequency=1,max_token_length=(num_bins//2)+1)

    # 训练模型
    tokenizer.train_from_iterator([symbol_str], trainer)

    # 保存模型
    tokenizer.save("bpe-tokenizer.json")

    # 编码
    encoded = tokenizer.encode(symbol_str)
    print(f"Encoded symbols: {encoded.tokens}")

    # 解码
    decoded = tokenizer.decode(encoded.ids,skip_special_tokens=False)

    print(f"Decoded symbols=={decoded}")
    print(len(decoded))

    # 将解码后的符号序列转换回原始形式
    decoded_symbols = convert_to_discretized(decoded,unicode2value)
    print(len(decoded_symbols))
    print(f"Decoded symbols (list): {decoded_symbols}")

    # 将解码后的符号序列转换回原始形式
    # decoded_symbols = list(map(int, decoded.split()))
    # print(f"Decoded symbols (list): {decoded_symbols}")




    # 还原时间序列数据
    restored_series = restore_series(decoded_symbols, np.linspace(min(time_series), max(time_series), num_bins + 1))
    print(f"Restored time series: {restored_series}")
    print(len(restored_series))

    plt.subplot(2, 2, 2)
    plt.plot(timestamp, restored_series)

    plt.show()

