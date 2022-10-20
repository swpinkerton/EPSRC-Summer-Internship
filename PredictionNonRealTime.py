from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.externals import joblib
from scipy import signal
import pandas

oldlen = 0
#windowsizer
length = 8312
extra = 0
#keep
overlap = 6652
#change
overlapRemander = 1660
li = []

df = pd.read_csv("Datasets/full20.csv", header=None)

if df:
    # df = df.drop([32])
    # df = df.melt()
    # df = df.replace(to_replace=r'\)', value='', regex=True)
    # df = df.replace(to_replace=r'\(', value='', regex=True)
    # df = df.replace(to_replace=r'None', value='', regex=True)
    # df = df['value'].str.split(',', expand=True)
    # df = df.dropna()
    #
    # df = df.astype(str).astype(float)
    # df = df[0] + df[1] * 1j
    df = df.to_numpy()
    df = np.abs(df)

    shape = int(df.shape[0])

    reshape = shape / 51

    df = np.reshape(df, (reshape, 51))

    b, A = signal.butter(1, 0.05)  # default is low pass
    Y = signal.filtfilt(b, A, df)

    df = pandas.DataFrame(Y).T
    df = df.mean(axis=0)
    df_max = df.max(axis=0)
    df_min = df.min(axis=0)
    df_std = df.std(axis=0)
    df_dif = df_max - df_min
    df = df.to_frame()
    df = df.T

    # df = pd.DataFrame(df).T
    dataset = df
    plt.close()
    dataset1 = df.T
    # dataset1 = dataset1.plot(figsize=(3,2),legend=None)

    if df_dif > 4:
        # if df_std > 0.39:
        pipe = joblib.load('Models//full.pkl')
        pred = pd.Series(pipe.predict(dataset))
        prediction = list(pred)

        output = prediction[0]
    else:
        output = "No Movement"

    print output
    li.append(output)
print(li)