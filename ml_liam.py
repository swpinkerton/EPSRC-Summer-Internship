import numpy as np
import time
import pandas
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
import glob
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification



# filename = 'C:\Users\swpin\OneDrive - University of Glasgow\Documents\University - use this\Glasgow\Intership2022\EPSRC Scott Fall Detection - Copy (2)/Datasets/1000full5000.csv'
filename = 'C:\Users\swpin\OneDrive - University of Glasgow\Documents\University - use this\Glasgow\Intership2022\EPSRC Scott Fall Detection - Copy (2)/Datasets/full480.csv'
CSV = pandas.read_csv(filename, header=None)




Array = CSV.values
X = Array[:,1:999999]
y = Array[:,0]


imp = SimpleImputer(missing_values=np.nan, strategy='mean')#, fill_value=0)
X = imp.fit_transform(X.T).T


print filename

def ml():
    # tic = time.time()
    # print '\n'
    # print 'Random Forest'
    # global rfc
    # rfc = RandomForestClassifier(n_estimators=200)
    # y_pred = cross_val_predict(rfc, X, y, cv=10)
    # print 'Confusion Matrix'
    # print confusion_matrix(y, y_pred)
    # print(classification_report(y, y_pred))
    # print 'Random Forest Accuracy', (accuracy_score(y, y_pred)*100),'%'
    # toc = time.time()
    # print('time taken : ' + str(toc - tic) + ' seconds')
    # print '\n'

    # tic = time.time()
    # print 'Bagged Trees'
    # global clf
    # clf = BaggingClassifier(base_estimator=SVC(gamma='scale',kernel='linear'),n_estimators=200)
    # y_pred = cross_val_predict(clf, X, y, cv=10)
    # print 'Confusion Matrix'
    # print confusion_matrix(y, y_pred)
    # print(classification_report(y, y_pred))
    # print 'Bagged Trees', (accuracy_score(y, y_pred)*100),'%'
    # toc = time.time()
    # print('time taken : ' + str(toc - tic) + ' seconds')
    # print '\n'

    # tic = time.time()
    # print 'KNN'
    # global KNN
    # KNN = KNeighborsClassifier(n_neighbors=3)
    # y_pred = cross_val_predict(KNN, X, y, cv=10)
    # print 'Confusion Matrix'
    # print confusion_matrix(y, y_pred)
    # print(classification_report(y, y_pred))
    # print 'KNN Accuracy', (accuracy_score(y, y_pred)*100),'%'
    # toc = time.time()
    # print('time taken : ' + str(toc - tic) + ' seconds')
    # print '\n'

    # tic = time.time()
    # print 'ADA'
    # global ADA
    # ADA = AdaBoostClassifier()
    # y_pred = cross_val_predict(ADA, X, y, cv=10)
    # print 'Confusion Matrix'
    # print confusion_matrix(y, y_pred)
    # print(classification_report(y, y_pred))
    # print 'KNN Accuracy', (accuracy_score(y, y_pred)*100),'%'
    # toc = time.time()
    # print('time taken : ' + str(toc - tic) + ' seconds')
    # print '\n'

    # tic = time.time()
    # print 'SVM Linear'
    # global SVM
    # SVM = svm.SVC(gamma='scale')
    # y_pred = cross_val_predict(SVM, X, y, cv=10)
    # print 'Confusion Matrix'
    # print confusion_matrix(y, y_pred)
    # print(classification_report(y, y_pred))
    # print 'SVM Accuracy', (accuracy_score(y, y_pred) * 100), '%'
    # toc = time.time()
    # print('time taken : ' + str(toc - tic) + ' seconds')
    # print '\n'

    tic = time.time()
    print 'SVM RBF'
    global SVMRBF
    SVMRBF = svm.SVC(gamma='auto', kernel='rbf', C=6.7)
    y_pred = cross_val_predict(SVMRBF, X, y, cv=10)
    print 'Confusion Matrix'
    print confusion_matrix(y, y_pred)
    print(classification_report(y, y_pred))
    print 'SVM Accuracy', (accuracy_score(y, y_pred)*100),'%'
    toc = time.time()
    print('time taken : ' + str(toc - tic) + ' seconds')
    print '\n'

    # tic = time.time()
    # print 'SVM RBF'
    # global SVMRBF
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # SVMRBF = SVC(gamma='auto', kernel='rbf', C=6.7)
    # SVMRBF.fit(X_train, y_train)
    # y_pred = SVMRBF.predict(X_test)
    # print 'Confusion Matrix'
    # print confusion_matrix(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    # acc = (accuracy_score(y_test, y_pred) * 100)
    # print 'SVM Accuracy', (accuracy_score(y_test, y_pred) * 100), '%'
    # toc = time.time()
    # print('time taken : ' + str(toc - tic) + ' seconds')
    # print '\n'

    # print 'LOGISTIC REGRESSION'
    # global lr
    # lr = LogisticRegression(solver='lbfgs', max_iter=2000)
    # y_pred = cross_val_predict(lr, X, y, cv=10)
    # print 'Confusion Matrix'
    # print confusion_matrix(y, y_pred)
    # print(classification_report(y, y_pred))
    # print 'LOGISTIC REGRESSION Accuracy', (accuracy_score(y, y_pred)*100),'%'
    # toc = time.time()
    # print('time taken : ' + str(toc - tic) + ' seconds')
    # print '\n'
    #
    # tic = time.time()
    # print 'NORMAL LINEAR DISCRIMINANT ANALYSIS'
    # lda = LinearDiscriminantAnalysis()
    # y_pred = cross_val_predict(lda, X, y, cv=10)
    # print 'Confusion Matrix'
    # print confusion_matrix(y, y_pred)
    # print(classification_report(y, y_pred))
    # print 'NORMAL LINEAR DISCRIMINANT ANALYSIS Accuracy', (accuracy_score(y, y_pred)*100),'%'
    # toc = time.time()
    # print('time taken : ' + str(toc - tic) + ' seconds')
    # print '\n'

def build_model():
    SVMRBF.fit(X, y)
    model_data = pickle.dumps(SVMRBF)
    model_file = "models/full.pkl"
    with open(model_file, "wb") as f:
        f.write(model_data)


ml()
build_model()


