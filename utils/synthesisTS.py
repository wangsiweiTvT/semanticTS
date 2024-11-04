import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import matplotlib.dates as mdates



def synthesis(L,k,T,A):
    r"""
    :param L: length of TS
    :param k: dec incr of TS
    :param T: list of temp
    :param A: list of amplitude
    :return:
    """
    timestamp=list(range(0,L,1))
    trend_value = list()
    season_value = list()
    for i in range(0,L):
        trend_value.append(i*k)
        season_tmp = 0
        for j in range(0,len(T)):
            season_tmp = season_tmp + (A[j]*math.sin((2*math.pi/T[j])*i))
        season_value.append(season_tmp)

    mix = [x + y for x, y in zip(season_value, trend_value)]
    return timestamp,mix,season_value,trend_value

def fft(ts):
    freqs = np.fft.fftfreq(len(ts), 1/len(ts))

    # 1. 快速傅里叶变换（FFT）
    result = np.fft.fft(ts)
    magnitude = np.abs(result)  # 幅度谱

    print(freqs)
    print(result)
    print(magnitude)
    return result,freqs,magnitude

if __name__ == '__main__':
    time,ts,s,t = synthesis(1000,0.001,[50,100,200,400],[1,1,2,2])
    plt.plot(time, ts, color='blue',linewidth=1)
    plt.plot(time, s, color='r',linewidth=1)
    plt.plot(time, t, color='g',linewidth=1)
    plt.show()

    result,f,a = fft(ts)

    plt.plot(f[:len(f) // 2], a[:len(a) // 2])
    plt.title("freq domain")
    plt.xlabel("f(Hz)")
    plt.ylabel("magnitude")
    plt.show()

    freq_idx = np.where(np.abs(f) <= 1)[0]
    fft_freqs_cp = np.zeros_like(result)
    fft_freqs_cp[freq_idx]=result[freq_idx]
    result[freq_idx] = 0

    # 2. 逆快速傅里叶变换（IFFT）
    ifft_result = np.fft.ifft(result)
    ifft_real = np.real(ifft_result)  # 提取逆变换后的实部（忽略小的虚数部分）
    plt.plot(time, ifft_real)
    plt.title("IFFT")
    plt.xlabel("t(s)")
    plt.ylabel("value")

    plt.show()