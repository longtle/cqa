'''
Created on Mar 25, 2015

@author: longtle
'''

import sys
from networkx.algorithms.shortest_paths.unweighted import predecessor
sys.path.insert(0, '../util/')
import Util
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats

from sklearn import svm, preprocessing, cross_validation, linear_model
from sklearn.ensemble import RandomForestClassifier

def readDataRegression(inputFile1, inputFile2, yFile1, yFile2):
    Xtrain1 = np.loadtxt(fname=inputFile1, dtype = float, delimiter = ',', skiprows = 1, usecols =(1,2,3,4,5,5,7,8,9,10,11))
    Xtrain2 = np.loadtxt(fname=inputFile2, dtype = float, delimiter = ',', skiprows = 1, usecols =(1,2,3,4,5,5,7,8,9,10,11))
    
    nTrain1, _ = Xtrain1.shape
    nTrain2, _ = Xtrain2.shape
    
    Ytrain1 = np.loadtxt(fname=yFile1, dtype = float, delimiter = ',', skiprows = 1, usecols =(1,))
    Ytrain2 = np.loadtxt(fname=yFile2, dtype = float, delimiter = ',', skiprows = 1, usecols =(1,))
    #print Xtrain1.shape, Xtrain2.shape
    print Ytrain1.shape, Ytrain1
    
    Xtrain = np.vstack((Xtrain1, Xtrain2))
    Ytrain = np.hstack((Ytrain1, Ytrain2))
    #print Xtrain
    #print Ytrain
    print 'In read data: ', Xtrain.shape, Ytrain.shape
    
    
    return Xtrain, Ytrain

def readDataRegressionOneClass(inputFile, yFile):
    Xtrain = np.loadtxt(fname=inputFile, dtype = float, delimiter = ',', skiprows = 1, usecols =(1,2,3,4,5,5,7,8,9,10,11))
    listId = np.loadtxt(fname = inputFile, dtype = str, delimiter = ',', skiprows = 1, usecols =(0,))
    
    yDict = Util.readDict(yFile, keyIndex = 0, valueIndex = 1, header = True, separator = ',')
    
    Ytrain = []
    for uid in listId:
        if uid in yDict:
            Ytrain.append(float(yDict[uid]))
    
    #Ytrain = [yDict[id] for id in listId]
    Ytrain = np.asarray(Ytrain)
    print len(Ytrain)
    
    print 'In read data: ', Xtrain.shape, len(Ytrain)
    
    
    return Xtrain, Ytrain

def regression(X, Y, algo):
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)
    
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, test_size=0.1, random_state = 0)
    Ytrain
    print 'Train: ', Xtrain.shape, Ytest.shape
    print 'Test: ', Xtest.shape, Ytest.shape
    #print Xtrain, Ytrain
    
    if (algo == 'linear'):
        reg = linear_model.LinearRegression(normalize=True)
        reg = reg.fit(Xtrain, Ytrain)
        pred = reg.predict(Xtest)
    
    elif (algo == 'lasso'):
        reg = linear_model.Lasso(normalize=True)
        reg = reg.fit(Xtrain, Ytrain)
        pred = reg.predict(Xtest)
    elif (algo == 'ridge'):
        reg = linear_model.Ridge(normalize=True)
        reg = reg.fit(Xtrain, Ytrain)
        pred = reg.predict(Xtest)
    
    print 'Score: ', reg.score(Xtest,Ytest)    
    #print 'Prediction: ', pred
    return pred

def regressionSep(Xtrain, Ytrain, Xtest, Ytest, algo):
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    Xtrain = imp.fit_transform(Xtrain)
    Xtest = imp.fit_transform(Xtest)
    
    print 'Train: ', Xtrain.shape, Ytest.shape
    print 'Test: ', Xtest.shape, Ytest.shape
    #print Xtrain, Ytrain
    
    if (algo == 'linear'):
        reg = linear_model.LinearRegression(normalize=True)
        reg = reg.fit(Xtrain, Ytrain)
        pred = reg.predict(Xtest)
    
    elif (algo == 'lasso'):
        reg = linear_model.Lasso(normalize=True)
        reg = reg.fit(Xtrain, Ytrain)
        pred = reg.predict(Xtest)
    elif (algo == 'ridge'):
        reg = linear_model.Ridge(normalize=True)
        reg = reg.fit(Xtrain, Ytrain)
        pred = reg.predict(Xtest)
    
    print 'Score: ', reg.score(Xtest,Ytest)    
    #print 'Prediction: ', pred
    return pred

def computeKendalTau(list1, list2):
    '''
    Given 2 list, compute the Kendall Tau correlation
    '''
    kt = scipy.stats.kendalltau(list1, list2)[0]
    print 'kt:', kt
    return kt

def computeKendalTauValue(y, yPred):
    l1 = [i[0] for i in sorted(enumerate(y), key=lambda x:x[1])]
    l2 = [i[0] for i in sorted(enumerate(yPred), key=lambda x:x[1])]
    print 'l1: ', l1
    print 'l2: ', l2
    return computeKendalTau(l1, l2)

def calcKTFromFile(inputFile, index1, index2, threshold):
    print 'Calculate KT from: ', inputFile
    inputFile = open(inputFile)
    rank1 = []
    rank2 = []
    #ignore the first line
    line = inputFile.readline()
    while 1:
        line = inputFile.readline()
        #print 'line: ', line
        if not line:
            break
        line = line.strip()
        token = line.split('\t')
        v1 = int (token[index1])
        v2 = int (token[index2])
        noOfRead1 = int(token[2])
        if (noOfRead1 >= threshold):
            rank1.append(v1)
            rank2.append(v2)
    #print 'rank1: ', rank1
    #print 'rank2: ', rank2
    print 'length: ', len(rank1)
    kt = scipy.stats.kendalltau(rank1, rank2)[0]
    print 'threshold ,kt:', threshold,  kt
    return kt

if __name__ == '__main__':
    myDir = '../../data/so2/1years/'
    logFile = myDir + 'correctness-rank.csv'
    csv_writer = csv.writer(open(logFile, 'wb'), delimiter = ',')
    
    for i in ['1w']:
    #for i in ['1w', '2w', '3w', '4w']:
    #for i in ['5posts', '10posts', '15posts', '20posts']:
        #X, Y = readDataClassify(myDir + i +'-user-top1p-feat.csv', myDir + i+'-user-other1p-feat.csv')
        #X, Y = readDataClassify(myDir + i +'-user-top10p-feat.csv', myDir + i +'-user-other10p-feat.csv')
    
        #X, Y = readDataRegression(myDir + i + '-user-top1p-feat.csv', myDir + i + '-user-other1p-feat.csv',
        #                    myDir +  '../' +'user-1years-only-top1p.csv', myDir + '../' + 'user-1years-only-other1p.csv')
        
        '''
        pred = regression(X, Y, algo='linear')
        for i in range (3):
            print 'Real, pred: ', Y[i], pred[i]
        '''
                          
        X, Y = readDataRegressionOneClass(myDir + i + '-user-top1p-feat.csv', myDir + '../' + 'user-1years-only-top1p.csv')
        Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, test_size=0.5, random_state = 0)
        pred = regressionSep(Xtrain, Ytrain, Xtest, Ytest, algo='ridge')
        print 'pred: ', pred
        print 'real value: ', Ytest
        
        computeKendalTauValue(Ytest, pred)
        
        #computeKendalTau([1,2,3], [1,2,2])
        
        #csv_writer.writerow([i, pred])
        
    pass