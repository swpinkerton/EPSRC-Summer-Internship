import pandas
import glob
import random
from sklearn.decomposition import PCA
import random
from scipy import signal

#path = r'//home/liam/host/Windows-to_from-Ubuntu/Data for Training/JWS 521 - Livingroom Setup Config2//'
path = r'//home/liam/host/Windows-to_from-Ubuntu/5G NT Extracted Data/Training//'


### For 5GNT
'''g1 = glob.glob(path + "/Activity/*.csv")              #In order to locate csv files, whose name may be unknown, the glob module is invoked  and glob method is called then. it returns all the csv files list located within the path
g2 = glob.glob(path + "/30 April/Activity/Enter/x*.csv")
g3 = glob.glob(path + "/30 April/Activity/Inactive Active/x*.csv")
g4 = glob.glob(path + "/Empty/*.csv")
g5 = glob.glob(path + "/01 May/Empty/x*.csv")
g6 = glob.glob(path + "/No Activity/*.csv")'''

### For Care hom Data
g1 = glob.glob(path + "/30 April/Activity/_active/*.csv")              #In order to locate csv files, whose name may be unknown, the glob module is invoked  and glob method is called then. it returns all the csv files list located within the path
g2 = glob.glob(path + "/30 April/Activity/Enter/*.csv")
g3 = glob.glob(path + "/30 April/Activity/Inactive Active/*.csv")
g4 = glob.glob(path + "/30 April/Empty/*.csv")
g5 = glob.glob(path + "/01 May/Empty/*.csv")
g6 = glob.glob(path + "/30 April/No Activity/*.csv")

print len(g1)
print len(g2)
print len(g3)
print len(g4)
print len(g5)
print len(g6)

li = []      # defining an empty list to store contents

pca = PCA(n_components=1)

# checking all the csv files in the specified path
for files in g1:
    # reading contents into data frame
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    df = df.iloc[[33]]
    
    

    b, a = signal.butter(1, 0.05) #default is low pass
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y)
    

    #label = 'Falling_Left'
    label = 'Activity'
    df.insert(0, 'Label', label)
    li.append(df)
########***********************************************************#########

for files in g2:
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    df = df.iloc[[33]]
    
    

    b, a = signal.butter(1, 0.05) #default is low pass
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y)
    

    #label = 'Falling_Right'
    label = 'Activity'
    df.insert(0, 'Label', label)
    li.append(df)
########***********************************************************#########

for files in g3:
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    df = df.iloc[[33]]
    
    

    b, a = signal.butter(1, 0.05) #default is low pass
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y) 

    #label = 'Falling_Forward'
    label = 'Activity'
    df.insert(0, 'Label', label)
    li.append(df)
########***********************************************************#########

for files in g4:
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    df = df.iloc[[33]]
    
    

    b, a = signal.butter(1, 0.05) #default is low pass
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y) 

    label = 'Empty Room'
    df.insert(0, 'Label', label)
    li.append(df)
########***********************************************************#########

for files in g5:
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    df = df.iloc[[33]]
    
    

    b, a = signal.butter(1, 0.05) #default is low pass
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y)

    #label = 'Walking_In_LR'
    label = 'Empty Room'
    df.insert(0, 'Label', label)
    li.append(df)
########***********************************************************#########

for files in g6:
    df = pandas.read_csv(files, index_col=False, header=None)
    df = df[df.columns[1:500000]]

    df = df.iloc[[33]]
    
    

    b, a = signal.butter(1, 0.05) #default is low pass
    y = signal.filtfilt(b, a, df)

    df = pandas.DataFrame(y)

    #label = 'Walking_In_LR'
    label = 'No Activity'
    df.insert(0, 'Label', label)
    li.append(df)
########***********************************************************#########


random.shuffle(li)
    
full = pandas.concat(li, axis=0, ignore_index=True,sort=False)
#full = full.sample(frac=1)
print full
#full.to_csv(index=False,path_or_buf='ZakirZone3_L1L2_Dataset.csv',header=False)
full.to_csv(index=False,path_or_buf='5GNT_CarehomeTrial1_30April_A_E_NA_01May_E_Dataset_Subcarrier34.csv',header=False)
print 'done'

#dataset_MultipleSubjects0to4AllClasses

#5GNT_Trial1_Activity_Dataset_Subcarrier40 -- Activity Dataset no PCA with only subcarrier 40
#5GNT_Trial1_Activity_Dataset              -- Activity Dataset no PCA with all subcarriers
#5GNT_Trial1_Activity_Dataset_PCA           -- Activity Dataset with PCA and all subcarriers
#5GNT_Trial1_Activity_Dataset_PCA_Subcarrier40 -- Activity Dataset with PCA and only subcarrier 40

## same for other classes