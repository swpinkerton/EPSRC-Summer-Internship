import pandas
from collections import Counter
import glob
from sklearn.decomposition import PCA
from scipy import signal
#from sklearn.externals
import joblib
from sklearn.impute import SimpleImputer
import numpy as np
import time
#import timing

pca = PCA(n_components=1)

g1 = glob.glob("/home/liam/host/Windows-to_from-Ubuntu/5G NT Extracted Data/Testing/01 May/Empty/*.csv")

filenames = []
Classiifcations = []
exec_time = []
nfeatures = []
#ttstart = []
#ttfinish = []

Model_n_features = 1238

for CSV in g1:
        filename = CSV.split('/')
        print('---------',filename)
        filename = filename[-1]
        filenames.append(filename)

        df = pandas.read_csv(CSV, index_col=False, header=None)
        del df[0]
        def preprocessing():
                global df
                df = df.iloc[[33]]
               
               
                ###PCA###
                #df = df.T
                #df = pca.fit_transform(df)
                #df = pandas.DataFrame(df)
                #df = df.T
               

                #df = df.add(100)

                ###Low Pass Filter###
                b, a = signal.butter(1, 0.05)
                y = signal.filtfilt(b, a, df)


                df = pandas.DataFrame(y)

               
                ###Average###
                #df = df.mean(axis=0)
                #df = df.to_frame()
                #df = df.T
               
               
                '''
                ###DWT###
                df = df.to_numpy()
                (df, cD) = pywt.dwt(df, 'db1')
                (df, cD) = pywt.dwt(df, 'db1')
                (df, cD) = pywt.dwt(df, 'db1')
                (df, cD) = pywt.dwt(df, 'db1')
                (df, cD) = pywt.dwt(df, 'db1')
                (df, cD) = pywt.dwt(df, 'db1')
                (df, cD) = pywt.dwt(df, 'db1')
                df = pandas.DataFrame(df)'''


        tstart = time.time()
        preprocessing()

        input_n_features = df.count(axis=1)[0]
        nfeatures.append(input_n_features)

        if input_n_features <= Model_n_features:
                new_col = Model_n_features - input_n_features
                nan = [None] * new_col
                df = df.reindex(columns = df.columns.tolist() + nan)
        elif input_n_features >= Model_n_features:
                dataset = df.iloc(axis=1)[:input_n_features]

        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        df = imp.fit_transform(df.T).T

        print df.shape

        pipe = joblib.load('models/5GNT_CarehomeTrial1_30April_A_E_NA_01May_E_Dataset_Subcarrier34.pkl')
        pred = pandas.Series(pipe.predict(df))
        prediction = list(pred)



        def most_frequent(List):
                occurence_count = Counter(List)
                return occurence_count.most_common(1)[0][0]

        list1 = prediction
        output = (most_frequent(list1))
        Classiifcations.append(output)
        #ttstart.append(tstart)
        #ttfinish.append(time.time())
        
dataframes = []
filenames = pandas.DataFrame(filenames)

dataframes.append(filenames)

#Classiifcations = pandas.DataFrame([Classiifcations,bClassification],columns=['Classiifcations','bClassification'])
Classiifcations = pandas.DataFrame({"Classiifcations": Classiifcations,"Features":nfeatures})

dataframes.append(Classiifcations)
#print Classiifcations

results = pandas.concat(dataframes, axis=1, ignore_index=True,sort=False)
results.columns = ['Filename', 'Classiifcations','Features']
results.to_csv(index=False,path_or_buf='results_18-12to18-28.csv',header=True)
print (results)