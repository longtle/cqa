'''
Created on Mar 16, 2015

@author: longtle
'''
import numpy as np
import matplotlib.pyplot as plt
import csv
import random

from sklearn import svm, preprocessing, cross_validation, linear_model, tree,\
    ensemble, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition.tests.test_nmf import random_state
from sklearn.metrics.metrics import confusion_matrix, f1_score


def readDataClassify(inputFile1, inputFile2):
    Xtrain1 = np.loadtxt(fname=inputFile1, dtype = float, delimiter = ',', skiprows = 1, usecols =(1,2,3,4,5,5,7,8,9,10,11))
    Xtrain2 = np.loadtxt(fname=inputFile2, dtype = float, delimiter = ',', skiprows = 1, usecols =(1,2,3,4,5,5,7,8,9,10,11))
    
    nTrain1, _ = Xtrain1.shape
    nTrain2, _ = Xtrain2.shape
    
    Ytrain1 = np.ones((nTrain1, 1), dtype = int)
    Ytrain2 = np.zeros((nTrain2, 1), dtype = int)
    #print Xtrain1.shape, Xtrain2.shape
    #print Ytrain1
    
    Xtrain = np.vstack((Xtrain1, Xtrain2))
    Ytrain = np.vstack((Ytrain1, Ytrain2))
    #print Xtrain
    #print Ytrain
    print 'In read data: ', Xtrain.shape, Ytrain.shape
    
    Ytrain = Ytrain[:,0]
    return Xtrain, Ytrain



def classify (X, Y, algo, logFile = None, featureSet = -1):
    #preprocessing NaA
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)
    
    #Select the group of features:
    if (featureSet == 0): #Personal features
        X = X[:, 0:3]
    elif (featureSet == 1):
        X = X[:, 3:7]
    elif (featureSet == 2):
        X = X[:, [7,9]]
    elif (featureSet == 3):
        X = X[:, [8, 10]]
    elif (featureSet == 4): #eliminate score
        X = X[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]]
    elif (featureSet == 5): #use score only
        X = X[:, [3]]
    print 'X: ', X.shape, X
    
    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, test_size=0.1, random_state = 0)
    
    print Xtrain.shape, Xtest.shape
    print Xtrain, Ytrain
    
    if (algo == 'svm'):
        clf = svm.SVC(kernel='linear', C=1)
        clf = clf.fit(Xtrain, Ytrain)
        
    elif (algo == 'ensemble'):
        clf = RandomForestClassifier(n_estimators=100, max_depth = None, min_samples_split=1, random_state=0)
        clf = clf.fit(Xtrain, Ytrain)
        featList = ['User-nPost','Post-type','length','Score','AnswerCount','CommentCount',
                    'FavoriteCount','Temporal-avg','std', 'avg-2nd', 'diff']
        importances = clf.feature_importances_
        print 'Feature importance: ', importances
        
        if (logFile != None):
            csv_writer = csv.writer(open(logFile, 'wb'), delimiter = ',')
            for i in range (len(featList)):
                print featList[i], importances[i]
                csv_writer.writerow([featList[i], importances[i]])
    elif (algo == 'logreg'):
        #http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
        clf = linear_model.LogisticRegression(C = 1)
        clf.fit(Xtrain, Ytrain)
    elif (algo == 'tree'):
        #http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
        clf = tree.DecisionTreeClassifier()
        clf.fit(Xtrain, Ytrain)
    elif (algo == 'ada-boost'):
        #http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
        clf = ensemble.AdaBoostClassifier()
        clf.fit(Xtrain, Ytrain)
    else:
        print 'invalid algorithm'
        
    print 'YTest:      ', Ytest[0:10]
    print 'YTest-Pred: ', clf.predict(Xtest)[0:10]
        
    score = clf.score(Xtest,Ytest)
    print 'score: ', score
    
    
    cm = confusion_matrix(Ytest, clf.predict(Xtest))
    print 'confusion matrix: ', cm
    
    '''
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    plt.show()
    '''
    
    print 'F1 score: ', f1_score(Ytest,clf.predict(Xtest))
    return score

def classifySubSet(X, Y, algo, ratio = 1, logFile = None, featureSet = -1):
    #ratio: the amount of subset data from 0 to 1
    
    nSample = X.shape[0]
    print '# of samples: ', nSample
    idx = [i for i in range (nSample)]
    topIdx = random.sample(idx, int(nSample*ratio))
    print 'Top Idx: ', topIdx
    
    newX = X[topIdx, :]
    newY = Y[topIdx]
    
    score = classify(newX, newY, algo)
    return score
    

def classifyBaseLine1(X, Y, logFile = None):
    #pre-processing NaA
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)
    
    #Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, test_size=0.1, random_state = 0)
    #print Xtrain.shape, Xtest.shape
    #print Xtrain, Ytrain
    print 'X: ', X.shape, Y.shape
    nExamples, _ = X.shape
    
    #check number of post
    nPost = X[:,0]
    avgScore = X[:, 3]
    
    score = nPost*avgScore
    print 'score: ', score
    
    #print '# posts: ', nPost
    
    idx = sorted(range(len(score)), key=lambda k: score[k])
    #print 'idx: ', idx
    
    nPositive = sum(Y)
    print 'nPositive: ', nPositive
    
    Ypred = [1 for i in range(nExamples)]
    for i in range (nExamples - nPositive):
        Ypred[idx[i]] = 0
    
    correct = [Ypred[i] == Y[i] for i in range (nExamples)]
    nCorrect = sum(correct)
    print 'nCorrect: ', nCorrect
    c = nCorrect/(nExamples*1.0)
    print 'c: ', c
    return c

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal Users', 'Top-1%'], rotation=45)
    plt.yticks(tick_marks, ['Normal Users', 'Top-1%'])
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':
    myDir = '../../data/so2/1years/'
    logFile = myDir + 'correctness.csv'
    csv_writer = csv.writer(open(logFile, 'wb'), delimiter = ',')
    
    #for i in ['1w', '2w', '3w', '4w']:
    for i in ['4w']:
    #for i in ['5posts', '10posts', '15posts', '20posts']:
    #for i in ['20posts']:
        X, Y = readDataClassify(myDir + i +'-user-top1p-feat.csv', myDir + i+'-user-other1p-feat.csv')
        #X, Y = readDataClassify(myDir + i +'-user-top10p-feat.csv', myDir + i +'-user-other10p-feat.csv')
    
        #score = classify(X, Y, algo = 'logreg', logFile=myDir+'feat-importances.csv')
        #score = classify(X, Y, algo = 'ensemble', logFile=myDir+'feat-importances.csv')        
        #score = classifyBaseLine1(X, Y)        
        #csv_writer.writerow([i, score])
        
        #for selectAlgo in ['logreg', 'ensemble', 'svm', 'tree', 'adaboost']:
        
        for  selectAlgo in ['svm']:
            score = classify(X, Y, algo = selectAlgo, logFile=myDir+'feat-importances.csv')
            csv_writer.writerow([i, selectAlgo, score])
        
        '''
        for j in [0, 1, 2, 3, -1]:
            score = classify(X, Y, algo = 'logreg', logFile=myDir+'feat-importances.csv', featureSet=j)
            csv_writer.writerow([j, score])
        '''
        ''' 
        for j in [4]:
            score = classify(X, Y, algo = 'logreg', logFile=myDir+'feat-importances.csv', featureSet=j)
            csv_writer.writerow([j, score])
        '''
        '''
        for j in [1, 0.5, 0.2, 0.1, 0.05, 0.01]:
            score = classifySubSet(X, Y, algo = 'logreg', ratio = j,logFile=myDir+'feat-importances.csv', featureSet=j)
            csv_writer.writerow([j, score])
        '''
    pass