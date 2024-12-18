import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import matplotlib.dates as mdates



def synthesis(L,k,T,A,sigma):
    r"""

    11.4 : 想要体现TS中的语义,借鉴 word2vec中的 idea
    word2vec idea : 一些离散的词,  A 后面 大概率是 B ,

    序列中,一段固定的确定的序列,无论从频域还是时域角度看,这段序列都已经确定,
    那么其中 子序列就产生固定搭配, 还是说 不同序列之间需要找固定搭配?

    比如 增长后必定上升这样的模式. 那么如何找到这样子的 最短子序列.
    可以借鉴 BPE（Byte Pair Encoding）能找到词根词缀

    那么其中任意的自序列都是确定的了.

    :param L: length of TS
    :param k: dec incr of TS
    :param T: list of temp
    :param A: list of amplitude
    :param sigma:
    :return:
    """
    timestamp=list(range(0,L,1))
    trend_value = list()
    season_value = list()
    for i in range(0,L):
        noise = np.random.normal(0, sigma)
        trend_value.append(i*k)
        season_tmp = 0
        for j in range(0,len(T)):
            season_tmp = season_tmp + (A[j]*math.sin((2*math.pi/T[j])*i))
        season_value.append(season_tmp+noise)

    mix = [x + y for x, y in zip(season_value, trend_value)]
    return timestamp,mix,season_value,trend_value

def fft(ts):
    freqs = np.fft.fftfreq(len(ts), 1/len(ts))

    # 1. 快速傅里叶变换（FFT）
    result = np.fft.fft(ts)
    magnitude = np.abs(result)  # 幅度谱
    return result,freqs,magnitude

def freq_filter(ts,freq):
    result, f, a = fft(ts)
    freq_idx = np.where(np.abs(f) > freq)[0]
    fft_freqs_cp = np.zeros_like(result)
    fft_freqs_cp[freq_idx] = result[freq_idx]
    result[freq_idx] = 0
    ifft_result = np.fft.ifft(result)
    ifft_real = np.real(ifft_result)  # 提取逆变换后的实部（忽略小的虚数部分）
    return ifft_real

if __name__ == '__main__':
    time,ts,s,t = synthesis(1000,0.001,[50,100,200,400],[1,1,2,2])
    plt.plot(time, ts, color='blue',linewidth=1)
    plt.plot(time, s, color='r',linewidth=1)
    plt.plot(time, t, color='g',linewidth=1)
    plt.show()

    ifft_real = freq_filter(ts,50)

    # plt.plot(f[:len(f) // 2], a[:len(a) // 2])
    # plt.title("freq domain")
    # plt.xlabel("f(Hz)")
    # plt.ylabel("magnitude")
    # plt.show()


    plt.plot(time, ifft_real)
    plt.title("IFFT")
    plt.xlabel("t(s)")
    plt.ylabel("value")

    plt.show()