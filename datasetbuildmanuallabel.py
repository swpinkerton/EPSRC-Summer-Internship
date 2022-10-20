##################
'''
This script builds a dataset by applying butterworth low pass filter and averaging all subcarriers
'''

import pandas
import glob
import random
from sklearn.decomposition import PCA
import random
from scipy import signal


path = r'/Data/'




### For Care hom Data
# g1 = glob.glob(path + "/40NormalWalk/*.csv")              #In order to locate csv files, whose name may be unknown, the glob module is invoked  and glob method is called then. it returns all the csv files list located within the path
# g2 = glob.glob(path + "/40LimpWalk/*.csv")
# g3 = glob.glob(path + "/40WobblyWalk/*.csv")
g1 = glob.glob(path + "/low/*.csv")              #In order to locate csv files, whose name may be unknown, the glob module is invoked  and glob method is called then. it returns all the csv files list located within the path
g2 = glob.glob(path + "/med/*.csv")
g3 = glob.glob(path + "/high/*.csv")

print len(g1)
print len(g2)
print len(g3)


li = []# defining an empty list to store contents
stdlist = []

pca = PCA(n_components=1)

# checking all the csv files in the specified path
for files in g1:
    # reading contents into data frame
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    b, a = signal.butter(1, 0.05) #default is low pass
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y)
    
    df = df.mean(axis=0)
    df_std = df.std()
    stdlist.append(df_std)
    df = df.to_frame() 
    df = df.T  

    #label = 'Falling_Left'
    # label = 'Walk1'
    label = 'NormalWalk'
    df.insert(0, 'Label', label)
    li.append(df)
########***********************************************************#########

for files in g2:
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    
    b, a = signal.butter(1, 0.05)
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y)
    
    df = df.mean(axis=0)
    df_std = df.std()
    stdlist.append(df_std)
    df = df.to_frame() 
    df = df.T  
    

    #label = 'Falling_Right'
    # label = 'Walk2'
    label = 'LimpWalk'
    df.insert(0, 'Label', label)
    li.append(df)
########***********************************************************#########

for files in g3:
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    
    b, a = signal.butter(1, 0.05)
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y)
    
    df = df.mean(axis=0)
    df_std = df.std()
    stdlist.append(df_std)
    df = df.to_frame() 
    df = df.T  

    #label = 'Falling_Forward'
    label = 'WobblyWalk'
    df.insert(0, 'Label', label)
    li.append(df)



########***********************************************************#########



print(stdlist)
random.shuffle(li)
    
full = pandas.concat(li, axis=0, ignore_index=True,sort=False)
print full
full.to_csv(index=False,path_or_buf='Datasets/fullstd.csv',header=False)
print 'done'
