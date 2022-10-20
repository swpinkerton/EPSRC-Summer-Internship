import subprocess
import sys
import pandas as pd
import numpy as np
from StringIO import StringIO
import matplotlib.pyplot as plt
import pause
import os
import time

pause.seconds(5)
#The value "y-x" is the number of files 
x = 1
y = 1

a = 50 #number on the file or the label

# label = 'day4scottNorm1'
# label = 'day4scottLimp1'
label = 'day4scottWobbly1'
#label = '3SubjectStanding'


while x <= y:

      output = subprocess.check_output([sys.executable, 'liamreceivertimerV2.py'])
      #output = subprocess.check_output([sys.executable, 'liam_ofdmremix14secs.py'])

      with open('outfile.txt', 'wb') as outfile:
            outfile.write(output)




      print 'data collected. Saving...'

      f = open('outfile.txt', 'r')

      liam = f.read()
      liam = StringIO(liam)
      data = pd.DataFrame(liam)

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


      data = data['a'].str.split(' ',expand=True).T



      data = data[6:58]
      df = data
      df = df.drop([32])


      df = df.melt()
      df = df.replace(to_replace=r'\)', value='', regex=True)
      df = df.replace(to_replace=r'\(', value='', regex=True)
      df = df['value'].str.split(',',expand=True)



      df = df.astype(str).astype(float)
      df = df[0] + df[1] * 1j
      df = df.to_numpy()
      df = np.abs(df)




      shape = int(df.shape[0])

      reshape = shape / 51

      df = np.reshape(df, (reshape, 51))



      dataset = pd.DataFrame(df).T
      dataset = dataset[dataset.columns[50:500000]]




      plt.close()
      dataset1 = dataset.T
      dataset1 = dataset1.plot(figsize=(5,3),legend=None,title=label + ' ' + str(a))


      a = str(a)

      stamp = time.time()
      stamp = str(stamp)
      stamp = stamp.split('.')
      stamp = stamp[0]

      try:
            os.mkdir('/home/liam/host/EPSRC Scott Fall Detection/Data/'+label)
      except:
            pass
      
      dataset1.figure.savefig('/home/liam/host/EPSRC Scott Fall Detection/Data/'+label+'/'+label+stamp+'-'+a+'.png')
      dataset.insert(0, 'Label', label)
      dataset.to_csv(path_or_buf='/home/liam/host/EPSRC Scott Fall Detection/Data/'+label+'/'+label+stamp+'-'+a+'.csv',index=False,header=False)     
      a = int(a)
      print dataset
      print a
      a = a + 1 
      x = x + 1
    
      
      

      plt.ion()
      plt.show()
      plt.pause(1)

from pydub import AudioSegment
from pydub.playback import play

song = AudioSegment.from_wav("sound.wav")
play(song)
  