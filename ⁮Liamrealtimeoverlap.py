from pylab import *
import numpy as np
from StringIO import StringIO
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import subprocess
import pause
import re
import sys
from pymongo import MongoClient
import pprint
import numpy as np
from StringIO import StringIO
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import subprocess
import pause
import re
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from collections import Counter 
from drawnow import drawnow
from scipy import signal
import pandas

pause.seconds(2)

label = 'liam'



try:
    os.remove("rawcsi.txt")
except:
    pass
open("rawcsi.txt", "w")

#Set Python script to run in background
os.system("xterm -e \"python liamreceiverV2.py > rawcsi.txt\" &")

#Set y as number of samples to be taken
x = 1
y = 5

#Set a as the numberfile to be used for saving files
a = 1




oldlen = 0
#windowsizer
length = 8312
extra = 0
#keep
overlap = 6652
#change
overlapRemander = 1660

newlen = []

while x <= y:
    
    
    li = []
    # arr = np.array([])


    f = open('rawcsi.txt', 'r')
    for line in f.readlines():
        if line:
            if re.search('ofdm_sync_chan_taps', line, re.I):
                # np.append(li,line)
                li.append(line)

    size_of_file =  len(li)
    # arr = np.array(li)
    
    while size_of_file >= length:
        
        
        print 'size_of_file', size_of_file
        extra = size_of_file - length
        DataWindow = size_of_file - extra
        print 'Data Window',DataWindow
        li = li[:-1]
        # arr = arr[:-1]
        print 'extra',extra
        

        data = li[oldlen:DataWindow]
        print 'data start:', oldlen,'end:',DataWindow
        oldlen = DataWindow - overlap
        length = length + overlapRemander

        print ''

        if data:
            
            
            data = pd.DataFrame(data)
 
            #data = data.iloc[::2]
            

            data.columns = ["a"]
            

            patternDel = "Offset"
            filter = data['a'].str.contains(patternDel)

            data = data[filter]


            patternDel = "\(.*\)"
            filter = data['a'].str.contains(patternDel)

            data = data[filter]


            data[['a','b']] = data['a'].str.split('#',expand=True)


            data = pd.DataFrame(data['b'])
            data[['a','b']] = data['b'].str.split('\[',expand=True)
            data[['a','b']] = data['b'].str.split('\]',expand=True)
            data = pd.DataFrame(data['a'])
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

            reshape = shape / 51

            df = np.reshape(df, (reshape, 51))
            
            b, A = signal.butter(1, 0.05) #default is low pass
            Y = signal.filtfilt(b, A, df)

            df = pandas.DataFrame(Y).T
            df = df.mean(axis=0)
            df_max = df.max(axis=0)
            df_min = df.min(axis=0)
            df_std = df.std(axis=0)
            df_dif = df_max - df_min
            df = df.to_frame() 
            df = df.T  


            #df = pd.DataFrame(df).T
            dataset = df
            #dataset = dataset.mean(axis=0)
            #dataset = dataset.to_frame()
            #dataset = dataset.T
            
            #dataset1 = dataset



            #print dataset
            #dataset2 = dataset1
            
            
            
            #print dataset

            plt.close()
            dataset1 = df.T
            #dataset1 = dataset1.plot(figsize=(3,2),legend=None)


            if df_dif > 4 :
            # if df_std > 0.39:
                pipe = joblib.load('Models//full.pkl')
                pred = pd.Series(pipe.predict(dataset))
                prediction = list(pred)

                output = prediction[0]
            else:
                output = "No Movement"

            print output


            a = str(a)
			#used to save figure(can be amended to include directories)
            #dataset1.figure.savefig('CSVs//'+label+a+'.png')
			#used to save CSV(can be amended to include directories)
            #dataset.to_csv(path_or_buf='CSVs//'+label+a+'.csv',index=False,header=False)
            
            #plt.ion()
            #plt.show()
            #plt.pause(0.0001)
            
            
            a = int(a)
            a = a + 1
        x = x 
        
    #pause.seconds(1)




            

            
        
