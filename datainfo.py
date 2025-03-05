import matplotlib.pyplot as plt
import pandas as pd
import glob
from util.synthesisTS import freq_filter
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.ensemble import RandomForestClassifier
from util.synthesisTS import freq_filter,fft


model = SVC(kernel='rbf', gamma=0.7, C=1.0)


if __name__ == '__main__':

    df0 = pd.read_csv('/Users/wangsiwei/dataset/BJTU_RAO_Bogie_Datasets/M0_G0_LA0_RA0/Sample_1/data_leftaxlebox_M0_G0_LA0_RA0_20Hz_0kN.csv')
    dfLA1 = pd.read_csv('/Users/wangsiwei/dataset/BJTU_RAO_Bogie_Datasets/M0_G0_LA1_RA0/Sample_1/data_leftaxlebox_M0_G0_LA1_RA0_20Hz_0kN.csv')
    dfLA2 = pd.read_csv('/Users/wangsiwei/dataset/BJTU_RAO_Bogie_Datasets/M0_G0_LA2_RA0/Sample_1/data_leftaxlebox_M0_G0_LA2_RA0_20Hz_0kN.csv')
    dfLA3 = pd.read_csv('/Users/wangsiwei/dataset/BJTU_RAO_Bogie_Datasets/M0_G0_LA3_RA0/Sample_1/data_leftaxlebox_M0_G0_LA3_RA0_20Hz_0kN.csv')
    dfLA4 = pd.read_csv('/Users/wangsiwei/dataset/BJTU_RAO_Bogie_Datasets/M0_G0_LA4_RA0/Sample_1/data_leftaxlebox_M0_G0_LA4_RA0_20Hz_0kN.csv')


    dfLA1LA2 = pd.read_csv('/Users/wangsiwei/dataset/BJTU_RAO_Bogie_Datasets/M0_G0_LA1+LA2_RA0/Sample_1/data_leftaxlebox_M0_G0_LA1+LA2_RA0_20Hz_0kN.csv')



    dfLA2LA3 = pd.read_csv('/Users/wangsiwei/dataset/BJTU_RAO_Bogie_Datasets/M0_G0_LA2+LA3_RA0/Sample_1/data_leftaxlebox_M0_G0_LA2+LA3_RA0_20Hz_0kN.csv')
    dfLA2LA4 = pd.read_csv('/Users/wangsiwei/dataset/BJTU_RAO_Bogie_Datasets/M0_G0_LA2+LA4_RA0/Sample_1/data_leftaxlebox_M0_G0_LA2+LA4_RA0_20Hz_0kN.csv')

    # df0_CH17 = df0['CH17']
    # dfLA1_CH17 = dfLA1['CH17']
    # dfLA2_CH17 = dfLA2['CH17']
    # dfLA1LA2_CH17 = dfLA1LA2['CH17']

    df0_CH17 = df0['CH17'][:100000]
    dfLA1_CH17 = dfLA1['CH17'][:100000]
    dfLA2_CH17 = dfLA2['CH17'][:100000]
    dfLA3_CH17 = dfLA3['CH17'][:100000]
    dfLA4_CH17 = dfLA4['CH17'][:100000]
    dfLA1LA2_CH17 = dfLA1LA2['CH17'][:100000]
    dfLA2LA3_CH17 = dfLA2LA3['CH17'][:100000]
    dfLA2LA4_CH17 = dfLA2LA4['CH17'][:100000]

    # print(CH11.size)
    timestamp = list(range(0, dfLA2_CH17.size, 1))
    plt.subplot(4, 2, 1)
    plt.title('LA2')
    plt.plot(timestamp, dfLA2_CH17,color = 'green')

    plt.subplot(4, 2, 2)
    plt.title('LA0')
    plt.plot(timestamp, df0_CH17,color = 'green')

    # m, f, a = fft(df0_CH17)
    # plt.plot(f[:len(f) // 2], a[:len(a) // 2])
    # plt.title("LA0 freq")
    # plt.xlabel("f(Hz)")
    # plt.ylabel("magnitude")

    plt.subplot(4, 2, 3)
    plt.title('LA2LA1')
    plt.plot(timestamp, dfLA1LA2_CH17,color = 'yellow')
    plt.subplot(4, 2, 4)
    plt.title('LA1')
    plt.plot(timestamp, dfLA1_CH17,color = 'yellow')
    # m, f, a = fft(dfLA1_CH17)
    # plt.plot(f[:len(f) // 2], a[:len(a) // 2])
    # plt.title("LA1 freq")
    # plt.xlabel("f(Hz)")
    # plt.ylabel("magnitude")

    plt.subplot(4, 2, 5)
    plt.title('LA2LA3')
    plt.plot(timestamp, dfLA2LA3_CH17,color = 'blue')
    plt.subplot(4, 2, 6)
    plt.title('LA3')
    plt.plot(timestamp, dfLA3_CH17,color = 'blue')
    # m, f, a = fft(dfLA2_CH17)
    # plt.plot(f[:len(f) // 2], a[:len(a) // 2])
    # plt.title("LA2 freq")
    # plt.xlabel("f(Hz)")
    # plt.ylabel("magnitude")
    # plt.subplot(4, 2, 2)
    # plt.title('LA1+LA2')
    # plt.plot(timestamp, dfLA1_CH17+dfLA2_CH17,color = 'red')

    plt.subplot(4, 2, 7)
    plt.title('LA2LA4')
    plt.plot(timestamp, dfLA2LA4_CH17,color = 'red')
    plt.subplot(4, 2, 8)
    plt.title('LA4')
    plt.plot(timestamp, dfLA4_CH17,color = 'red')
    # m, f, a = fft(dfLA1LA2_CH17)
    # plt.plot(f[:len(f) // 2], a[:len(a) // 2])
    # plt.title("LA2_LA2 freq")
    # plt.xlabel("f(Hz)")
    # plt.ylabel("magnitude")

    plt.tight_layout()
    plt.show()
    # print()



    #
    # files = glob.glob('2018aiops_dataset/*')
    # timestamp = list(range(0, 800, 1))
    # num = 1
    # for file in files:
    #     # ,3,5,7,8,9,12,13,18,21,2317
    #     if num in (3, 7, 9, 12, 13, 18, 21, 23):
    #         print(file)
    #         df = pd.read_csv(file)
    #         # (df - df.min()) / (df.max() - df.min())
    #         values = df.head(800)['value']
    #         max_value = values.max()
    #         min_value = values.min()
    #         values = (values - min_value) / (max_value - min_value)
    #         plt.subplot(2, 2, 1)
    #
    #         plt.plot(timestamp, values)
    #
    #         ifft_real = freq_filter(values,40)
    #
    #         # plt.subplot(2, 2, 2)
    #         # plt.plot(f[:len(f) // 2], a[:len(a) // 2])
    #         # plt.title("freq domain")
    #         # plt.xlabel("f(Hz)")
    #         # plt.ylabel("magnitude")
    #
    #
    #         plt.subplot(2, 2, 3)
    #         plt.plot(timestamp, ifft_real)
    #         plt.title("IFFT")
    #         plt.xlabel("t(s)")
    #         plt.ylabel("value")
    #
    #         plt.show()
    #
    #     num = num + 1



