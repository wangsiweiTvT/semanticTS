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





if __name__ == '__main__':
    df = pd.read_csv('/Users/wangsiwei/Desktop/训练数据group3.csv')
    df = df.sample(n=100000, random_state=42)

    df_x = df.iloc[:, :53]
    df_x_normalized = (df_x - df_x.min()) / (df_x.max() - df_x.min())
    df_x_normalized = df_x_normalized.fillna(0)
    # print(df_x_normalized.describe())

    # 特征和目标变量
    # X = df_x_normalized.iloc[:, :35]
    # X = df_x_normalized.iloc[:, 35:53]
    X = df_x_normalized
    y_class = df['leakPipeId']
    y_reg = df['leakage']
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

    # classifier = SVC(kernel='rbf', random_state=42)
    classifier = RandomForestClassifier(random_state=42)

    # 训练模型
    # classifier.fit(X_train, y_train)
    # 保存模型
    # joblib.dump(classifier, 'randomfore_18_10W.joblib')

    # 加载模型
    loaded_model = joblib.load('randomfore_18_10W.joblib')

    # 预测测试集
    y_pred = loaded_model.predict(X_test)

    # 计算准确度
    accuracy = accuracy_score(y_test, y_pred)

    print("测试集准确度:", accuracy)






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



