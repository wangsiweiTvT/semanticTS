import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import pandas as pd
import matplotlib.pyplot as plt
from synthesisTS import synthesis
import unicodedata
from synthesisTS import freq_filter

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

def restore_tokens(tokens:list,bins,if_diff:bool,first_element=0):
    first_value = first_element
    timestamp_end = 0
    timestamps = list()
    values = list()
    for index,token in enumerate(tokens):
        token_len = len(token)
        if(if_diff and index==0):timestamp_token = range(timestamp_end,timestamp_end+token_len+1)
        else:timestamp_token = range(timestamp_end,timestamp_end+token_len)
        timestamps.append(timestamp_token)
        timestamp_end = timestamp_end + len(timestamp_token)
        token_decoded_symbols = convert_to_discretized(token, unicode2value)
        restored_series = restore_series(token_decoded_symbols, bins)
        if(if_diff):
            original_token = np.cumsum(np.insert(restored_series, 0, first_value))
            first_value=original_token[-1]
            if(index==0):original_token = original_token
            else:original_token = original_token[1:]
            values.append(original_token)
        else : values.append(restored_series)
    return timestamps,values

if __name__ == '__main__':

    # ETT
    df = pd.read_csv('../2018aiops_dataset/54e8a140f6237526_32992.csv')
    time_series = df.head(800)['value']
    timestamp = list(range(0, 800, 1))
    # 合成
    # timestamp, time_series, s, t = synthesis(1000, 0.001, [25,50, 100, 200, 400], [1,1, 1, 1, 1])


    # 原序列
    plt.subplot(2, 2, 1)
    plt.plot(range(0,len(time_series)), time_series)

    # 去高频
    time_series = freq_filter(time_series,50)
    plt.subplot(2, 2, 2)
    plt.plot(range(0,len(time_series)), time_series)
    # plt.show()



    first_element = time_series[0]
    #差分
    df = pd.DataFrame(time_series, columns=['value'])
    df['first_diff'] = df['value'].diff()
    time_series = df['first_diff'][1:]


    num_bins = len(time_series)
    # 将时间序列数据离散化为索引 [478 478 501 503 485 465 440 431 395 366 333 329 348 387 368 275 229 222]
    symbols = discretize_series(time_series, num_bins)
    # print(f"Discretized symbols: {symbols}")

    value2unicode,unicode2value = create_value_to_unicode_map(symbols)
    # 将符号序列转换为字符串形式
    unicode_series = convert_to_unicode(symbols, value2unicode)
    # print(f"unicode_series: {unicode_series}")
    symbol_str = ''.join(unicode_series)

    # 初始化BPE模型
    BPE_model = models.BPE()
    tokenizer = Tokenizer(BPE_model)

    # 定义预处理器和训练器
    trainer = trainers.BpeTrainer(vocab_size=num_bins, min_frequency=1,max_token_length=500)

    # 训练模型
    tokenizer.train_from_iterator([symbol_str], trainer)

    # 保存模型
    tokenizer.save("bpe-tokenizer.json")

    # 编码
    encoded = tokenizer.encode(symbol_str)

    # 解码
    decoded = tokenizer.decode(encoded.ids,skip_special_tokens=False)

    tokens = decoded.split()

    decoded_symbols = convert_to_discretized(decoded,unicode2value)

    colors = ['red','green','blue','yellow']
    timestamp_end = 0

    plt.subplot(2, 2, 3)
    bins = np.linspace(min(time_series), max(time_series), num_bins + 1)
    time_list,origin_value_list =restore_tokens(if_diff=True,tokens=tokens,bins=bins,first_element=first_element)
    # time_list,origin_value_list =restore_tokens(if_diff=False,tokens=tokens,bins=bins)
    for index, time in enumerate(time_list):
        plt.plot(time, origin_value_list[index], color=colors[encoded.ids[index] % len(colors)])

    plt.show()
    # 还原时间序列数据
    # restored_series = restore_series(decoded_symbols, bins)
    # restored_series = restore_tokens(if_diff=True,tokens = [decoded_symbols], bins=bins,first_element=first_element)
    # restored_series = restore_tokens(if_diff=False,tokens = [decoded_symbols], bins=bins)
    # print(f"Restored time series: {restored_series}")
    # print(len(restored_series))

    # plt.subplot(2, 2, 4)
    # plt.plot(range(0,len(restored_series)), restored_series)





