from datetime import datetime
import datetime
import pandas
from collections import Counter
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import subprocess
import sys
import pandas as pd
import numpy as np
from StringIO import StringIO##python2
from io import StringIO
import joblib
from sklearn.impute import SimpleImputer
from collections import Counter
from pymongo import MongoClient
import glob
import os
#import urllib.parse
import ssl
from scipy import signal
import time
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#os.system("xterm -e \"python datacollection.py\" &")

path = ''
number_of_subcarriers = 51
model_name = 'models//JWS521+Empty427b_DataLivingRoomConfig2.pkl'

Model_n_features = open(model_name, "r")
Model_n_features = Model_n_features.readlines()
Model_n_features = Model_n_features[124]
Model_n_features = Model_n_features.strip('I')
Model_n_features = int(Model_n_features)

oldfile = ['file']

X = 1
Y = 2


while X < Y:
    now = datetime.datetime.now()
    now = str(now)
    now = now.replace(":", "-")
    date = now.split(" ")
    date = date[0]

    CSVTXT = open(date+".csv", "a")


    StartTime =time.time()
    
    
    files = glob.glob(path+"data//*.txt")
    files.sort(reverse=True)
    if files:
        files.pop(0)
    for file in files:
        os.remove(file)
    if files:
        filename = glob.glob(path+"data//*.txt")
        filename = filename[0]
    else:
        filename = 'file'

    if filename == oldfile[0]:
        print ('no Match')
    else:
        print ('Match')        

        ########################
        li = []
        try:

            f = open(filename, 'r')
            for line in f.readlines():
                    if line:
                            if re.search('ofdm_sync_chan_taps', line, re.I):
                                    li.append(line)
            li = li[:-1]

            try:

                data = li
                data = pandas.DataFrame(data)
                data.columns = ["a"]
                patternDel = "Offset"
                filter = data['a'].str.contains(patternDel)
                data = data[filter]
                patternDel = "\(.*\)"
                filter = data['a'].str.contains(patternDel)
                data = data[filter]
                data[['a','b']] = data['a'].str.split('#',expand=True)
                data = pandas.DataFrame(data['b'])
                data[['a','b']] = data['b'].str.split('\[',expand=True)
                data[['a','b']] = data['b'].str.split('\]',expand=True)
                data = pandas.DataFrame(data['a'])
                data = data[~data['a'].str.contains("Enter")]
                data = data['a'].str.split(' ',expand=True).T

                data = data[6:58]
                df = data
                df = df.drop([32])

                df = df.melt()
                df = df.replace(to_replace=r'\)', value='', regex=True)
                df = df.replace(to_replace=r'\(', value='', regex=True)
                df = df.replace(to_replace=r'None', value='', regex=True)
                df = df['value'].str.split(',',expand=True)
                df = df.dropna()
                df = df.astype(str).astype(float)
                df = df[0] + df[1] * 1j
                df = df.to_numpy()
                df = np.abs(df)
                shape = int(df.shape[0])
                reshape = shape / number_of_subcarriers
                df = np.reshape(df, (reshape, number_of_subcarriers))
                df = pandas.DataFrame(df).T    
                ########################


                #CSV = pandas.read_csv(filename, header=None)


                #df = CSV



                ####Preprocessing###
                #df = df.iloc[[39]]

                ###Low Pass Filter###
                b, a = signal.butter(1, 0.5)
                y = signal.filtfilt(b, a, df)
                df = pandas.DataFrame(y)

                ###Features###
                '''df['mean'] = df.mean(axis=1)
                df['max'] = df.max(axis=1)
                df['min'] = df.min(axis=1)
                df['kurtosis'] = df.kurtosis(axis=1)
                df['skew'] = df.skew(axis=1)
                df['std'] = df.std(axis=1)
                df = df.iloc[:,-6:]
                df = pandas.DataFrame(df)'''

                input_n_features = df.count(axis=1)[0]
                #nfeatures.append(input_n_features)


                if input_n_features <= Model_n_features:
                        new_col = Model_n_features - input_n_features
                        nan = [None] * new_col
                        df = df.reindex(columns = df.columns.tolist() + nan)
                elif input_n_features >= Model_n_features:
                        df = df.iloc(axis=1)[:Model_n_features]

                imp = SimpleImputer(missing_values=np.nan, strategy='mean')#'constant', fill_value=0)
                df = imp.fit_transform(df.T).T


                pipe = joblib.load(model_name)
                pred = pandas.Series(pipe.predict(df))
                prediction = list(pred)

                output = prediction[0]
                EndTime = time.time()

                Time = EndTime - StartTime
                Time = str(Time)
                CSVdata = (filename+','+output+','+Time+'\n')
                CSVTXT.write(CSVdata)
                CSVTXT.close()
            except:
                pass
        except:
            pass

    oldfile[0] = filename






