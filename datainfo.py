import matplotlib.pyplot as plt
import pandas as pd
import glob
from utils.synthesisTS import fft
import numpy as np



if __name__ == '__main__':

    files = glob.glob('2018aiops_dataset/*')
    timestamp = list(range(0, 800, 1))
    num = 1
    for file in files:
        # ,3,5,7,8,9,12,13,18,21,2317
        if num in (3, 7, 9, 12, 13, 18, 21, 23):
            df = pd.read_csv(file)
            # (df - df.min()) / (df.max() - df.min())
            values = df.head(800)['value']
            max_value = values.max()
            min_value = values.min()
            values = (values - min_value) / (max_value - min_value)
            plt.subplot(2, 2, 1)

            plt.plot(timestamp, values)

            result, f, a = fft(values)

            plt.subplot(2, 2, 2)
            plt.plot(f[:len(f) // 2], a[:len(a) // 2])
            plt.title("freq domain")
            plt.xlabel("f(Hz)")
            plt.ylabel("magnitude")

            freq_idx = np.where(np.abs(f) >= 50)[0]
            fft_freqs_cp = np.zeros_like(result)
            fft_freqs_cp[freq_idx] = result[freq_idx]
            result[freq_idx] = 0

            # 2. 逆快速傅里叶变换（IFFT）
            ifft_result = np.fft.ifft(result)
            ifft_real = np.real(ifft_result)  # 提取逆变换后的实部（忽略小的虚数部分）
            plt.subplot(2, 2, 3)
            plt.plot(timestamp, ifft_real)
            plt.title("IFFT")
            plt.xlabel("t(s)")
            plt.ylabel("value")

            plt.show()

        num = num + 1



