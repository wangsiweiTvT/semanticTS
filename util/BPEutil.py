import numpy as np
from tokenizers import Tokenizer, models, trainers
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from util.synthesisTS import freq_filter,fft,synthesis
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# 添加高斯噪声
def add_gaussian_noise(time_series, sigma):
    noise = np.random.normal(0, sigma, np.array(time_series).shape)
    noisy_time_series = time_series + noise
    return noisy_time_series

def create_value_to_unicode_map(discretized_data):
    unique_values = sorted(set(value for value in discretized_data))
    value2unicode = {value: chr(i+1000) for i, value in enumerate(unique_values)}
    unicode2value = {chr(i+1000): value for i, value in enumerate(unique_values)}
    return value2unicode,unicode2value

def convert_to_unicode(discretized_data, value2unicode):
    unicode_series = [value2unicode[value] for value in discretized_data]
    return unicode_series



# 将unicode 转成离散的idx ,进一步可以转化回真值       &$#*@ @$&$* $*& #@^&@ $#*^(#@^@#*&&(^$ -> 1,2,3,4,5
def convert_to_discretized(decoded, unicode2value):
    idx = list()
    for value in decoded:
        if value==' ':continue
        idx.append(unicode2value[value])
    return idx
# 离散化（将数值范围分成若干区间）
def discretize_series(series, num_bins):
    bins = np.linspace(min(series), max(series), num_bins )
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
    df = pd.read_csv('E:/ideaproj/ETDataset/ETT-small/ETTm1.csv')
    time_series1 = df['HUFL']
    time_series2 = df['HULL']
    time_series3 = df['MUFL']
    time_series4 = df['MULL']
    time_series5 = df['LUFL']
    time_series6 = df['LULL']
    time_series7 = df['OT']



    time_series = time_series2
    print(time_series.size)
    timestamp = list(range(0, time_series.size, 1))
    # 合成
    # x, ts, s, t = synthesis(100000, 0.000, [25,30, 40, 125, 150,500], [1,1, 1, 1, 1,1],0.0)
    # time_series=ts
    # timestamp=x
    # 原序列


    plt.subplot(2, 2, 1)
    plt.plot(range(0,len(time_series)), time_series)


    # plt.plot(range(0, len(time_series)), time_series1,color = 'red')
    # plt.plot(range(0, len(time_series)), time_series2,color = 'yellow')
    # plt.plot(range(0, len(time_series)), time_series3,color = 'blue')
    # plt.plot(range(0, len(time_series)), time_series4,color = 'green')
    # plt.plot(range(0, len(time_series)), time_series5,color = 'pink')
    # plt.plot(range(0, len(time_series)), time_series6,color = 'purple')
    # plt.plot(range(0, len(time_series)), time_series7,color = 'black')

    # plt.plot(range(0,len(time_series)), time_series)
    # plt.subplot(2, 2, 2)
    # m,f,a = fft(time_series)
    # plt.plot(f[:len(f) // 2], a[:len(a) // 2])
    # plt.title("freq domain")
    # plt.xlabel("f(Hz)")
    # plt.ylabel("magnitude")
    # plt.show()

    # 绘制ACF和PACF图
    # fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    # plot_acf(time_series_diff, lags=40, ax=axes[0])
    # plot_pacf(time_series_diff, lags=40, ax=axes[1])
    # plt.show()

    #ARIMA 去噪
    # arima_model = sm.tsa.ARIMA(time_series, order=(0, 1, 0)).fit()
    # time_series = arima_model.fittedvalues
    # plt.subplot(2, 2, 3)
    # plt.plot(range(0,len(time_series)), time_series)


    # 去高频
    # time_series = freq_filter(time_series,6000)

    first_element = time_series[0]
    #差分
    # df = pd.DataFrame(time_series, columns=['value'])
    # df['first_diff'] = df['value'].diff()
    time_series = time_series.diff()[1:]

    plt.subplot(2, 2, 2)
    plt.plot(range(0,len(time_series)), time_series)

    num_bins = 30000
    # 将时间序列数据离散化为索引 [478 478 501 503 485 465 440 431 395 366 333 329 348 387 368 275 229 222]
    symbols = discretize_series(time_series, num_bins)
    print(f"Discretized symbols: {symbols}")


    value2unicode,unicode2value = create_value_to_unicode_map(symbols)
    # 将符号序列转换为字符串形式
    unicode_series = convert_to_unicode(symbols, value2unicode)
    # print(f"unicode_series: {unicode_series}")
    symbol_str = ''.join(unicode_series)

    # 初始化BPE模型
    BPE_model = models.BPE()
    tokenizer = Tokenizer(BPE_model)

    # 定义预处理器和训练器
    trainer = trainers.BpeTrainer( min_frequency=2,max_token_length=1000)

    # 训练模型
    tokenizer.train_from_iterator([symbol_str], trainer)

    # 保存模型
    tokenizer.save("bpe-tokenizer.json")

    # 编码
    encoded = tokenizer.encode(symbol_str)

    # frequency = Counter(encoded.ids)
    # print(frequency)
    # vocab = tokenizer.get_vocab()


    # 解码
    decoded = tokenizer.decode(encoded.ids,skip_special_tokens=False)
    # print(decoded)
    tokens = decoded.split()
    print(len(tokens))
    # tsBPE2vec_model = Word2Vec(sentences=[tokens], vector_size=100, window=10, min_count=0, workers=4)
    # vocab = tsBPE2vec_model.wv.key_to_index.keys()
    decoded_symbols = convert_to_discretized(decoded,unicode2value)
    print(decoded_symbols)
    # print(tokens[0:10])
    # symbol_sub_str = ''.join(tokens[0:10])
    # sub_encoded = tokenizer.encode(symbol_sub_str)
    # sub_decoded = tokenizer.decode(sub_encoded.ids,skip_special_tokens=False)
    # print(sub_decoded)


    colors = ['red','green','blue','yellow']
    timestamp_end = 0

    plt.subplot(2, 2, 3)
    bins = np.linspace(min(time_series), max(time_series), num_bins+1 )
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


    # plt.plot(range(0,len(restored_series)), restored_series)

    # for token in vocab:
    #     if(len(token)>20):
    #         plt.subplot(2, 2, 4)
    #         time_list, origin_value_list = restore_tokens(if_diff=False, tokens=[token], bins=bins)
    #         for index, time in enumerate(time_list):
    #             plt.plot(time, origin_value_list[index],color='red')
    #         # 找到与某个词最相似的词
    #         similar_token = tsBPE2vec_model.wv.most_similar(token, topn=1)
    #         print("Most similar words to '",token,"':", similar_token)
    #         for word,score in similar_token:
    #             sim_time_list, sim_origin_value_list = restore_tokens(if_diff=False, tokens=[word], bins=bins)
    #             for index, time in enumerate(sim_time_list):
    #                 plt.plot(time, sim_origin_value_list[index])
    #         plt.show()















