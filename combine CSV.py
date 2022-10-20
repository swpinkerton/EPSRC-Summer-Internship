import pandas
import glob
import random
from sklearn.utils import shuffle


li = []

file1 = '5GNT_Trial1_NoActivity_Dataset_Subcarrier40.csv'
file1 = pandas.read_csv(file1, header=None)
li.append(file1)

file2 = '5GNT_Trial1_Empty_Dataset_Subcarrier40.csv'
file2 = pandas.read_csv(file2, header=None)
li.append(file2)

file3 = '5GNT_Trial1_Activity_Dataset_Subcarrier40.csv'
file3 = pandas.read_csv(file3, header=None)
li.append(file3)

#file4 = 'dataset_USRP_Paper_ZoneThreeAll.csv'
#file4 = pandas.read_csv(file4, header=None)
#li.append(file4)

#file5 = 'dataset_Entering_LivingRoom.csv'
#file5 = pandas.read_csv(file5, header=None)
#li.append(file5)

#file6 = 'dataset_SittingTZ.csv'
#file6 = pandas.read_csv(file6, header=None)
#li.append(file6)

#file7 = 'dataset_StandingTZ.csv'
#file7 = pandas.read_csv(file7, header=None)
#li.append(file7)

#file8 = 'dataset_WalkingTZ.csv'
#file8 = pandas.read_csv(file8, header=None)
#li.append(file8)



random.shuffle(li)

full = pandas.concat(li, axis=0, ignore_index=True,sort=False)
full = shuffle(full)
full.to_csv(index=False,path_or_buf='5GNT_Trial1_Emp_NoA_Activity_Dataset_Subcarrier40.csv',header=False)
print full