import sys
import math
import numpy as np
import csv
import time
import pandas as pd
import matplotlib.pylab as plt
from textstat.textstat import textstat

from sklearn.preprocessing import Imputer, label_binarize
from sklearn import svm, preprocessing, cross_validation, linear_model, tree, ensemble, metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.decomposition.tests.test_nmf import random_state
from sklearn.metrics.metrics import confusion_matrix, f1_score, roc_curve, auc

def classify(inputFile, algo = None,option = None, outputFile = None, logFile = None, predFile = None):
    '''
    option = 0: use all features
    =1: personal features
    =2: community features
    =3: textual features
    =4: contextual features
    '''
    allData = pd.read_csv(inputFile)
    print 'allData size: ', allData.shape#, allData.ix[0,:]
    Y = allData["outcome"]

    X = allData
    #delete the outcome from predictor X
    del X["outcome"]
    
    aid = allData["user_id"]
    del X["user_id"]
    print 'X size:', X.shape, allData.shape
    #del X["idx"]
    
    #select the set of features based on option
    if (option ==  None or option == 0):
        print 'Get all features'
    elif (option == 1):
        X = X[['u_grade', 'n_answers', 'life_days', 'count_logins']]
    elif (option == 2):
        #X = X[['degs', 'deg_adj', 'ego','CC','CC_adj','ego_out','ego_adj']]
        X = X[['degs', 'deg_adj', 'ego','CC','CC_adj','ego_out','ego_adj', 'del_ratio', 'n_best_ans']]
    elif (option == 3):
        X = X[['avg_length', 'contain_tex','well_format','ari', 'fres']]
    elif (option == 4):
        X = X[['avg_answer_time', 'typing_speed']]
    else:
        print 'Invalid option'
        return
    
    featList = list(X.columns.values)
    
    #print 'Xtest index value: ', X.index.tolist()
    
    #Convert to matrix:
    #X = X.as_matrix()
    
    #replace NaN value by mean    
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    X = imp.fit_transform(X) 
    print 'X shape: ', X.shape #, X
    
    #normalize data
    X = preprocessing.scale(X)
    print 'X[0]: ', X[0]
    
    #Y = Y.as_matrix()
    Y = Y.get_values().tolist()
    
    le = preprocessing.LabelEncoder()
    le.fit(['P', 'N'])
    Y = le.transform(Y)
    
    print 'Original Y: ', Y.shape, Y
    
    # Binarize the output, useful for plotting ROC
    '''
    Y = label_binarize(Y, classes=[0, 1])
    print 'New Y: ', Y.shape, Y
    nclass = Y.shape[1]
    '''
    indices = np.arange(X.shape[0])
    Xtrain, Xtest, Ytrain, Ytest, _, idx2 = cross_validation.train_test_split(X, Y, indices, test_size=0.3, random_state = 0)
    print Xtrain.shape, Xtest.shape
    print 'Xtrain :', Xtrain.shape#, Xtrain[0,:]
    print 'Ytrain:', Ytrain.shape
    #print 'idx2: ', idx2.shape, idx2
    
    if (algo == 'logreg'):
        #http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
        clf = linear_model.LogisticRegression(C = 1)
        clf.fit(Xtrain, Ytrain)
    elif (algo == 'tree'):
        #http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
        clf = tree.DecisionTreeClassifier()
        clf.fit(Xtrain, Ytrain)  
    elif (algo == 'ensemble'):
        start = time.time()
        clf = RandomForestClassifier(n_estimators=100, max_depth = None, min_samples_split=1, random_state=0)        
        clf = clf.fit(Xtrain, Ytrain)
        end = time.time()
        print 'Training time: ', (end-start)
        
        #lost the typing speed
        
        importances = clf.feature_importances_
        print 'Feature importance: ', importances
        
        if (logFile != None):
            csv_writer = csv.writer(open(logFile, 'wb'), delimiter = ',')
            for i in range (len(featList)):
                print featList[i], importances[i]
                csv_writer.writerow([featList[i], importances[i]])
        
    elif (algo == 'bagging'):
        start = time.time()
        clf = BaggingClassifier(n_estimators=100)
        clf = clf.fit(Xtrain, Ytrain)
        end = time.time()
        print 'Training time: ', (start - end)
                
    else:
        print 'Invalid option'
        return
    
        
    score = clf.score(Xtest,Ytest)    
    print 'score: ', score
    
    start = time.time()
    Ypred = clf.predict(Xtest)
    end = time.time()
    print 'Test time: ', (end -start)
    print 'Average test time:', 1.0*(start - end)/Ypred.size
    
    
    print 'Ypred: ', Ypred.shape, Ypred[0]
    
    f1Score = f1_score(Ytest, Ypred)
    print 'F1 score: ', f1Score
    
    Yscore = clf.predict_proba(Xtest)
    
    #cm = confusion_matrix(Ytest, Ypred, labels = [1, 0])
    cm = confusion_matrix(Ytest, Ypred, labels = [0,1])
    print 'confusion matrix: \n', cm
    
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    
    #write prediction value and real value:
    if (predFile != None):
        csv_writer_pred = csv.writer(open(predFile, 'wb'), delimiter = ',')
        #csv_writer_pred.writerow(['idx', 'answer_id'] +  featList + ['outcome', 'prediction', 'score'])
        csv_writer_pred.writerow(['answer_id','outcome', 'prediction', 'score'])
        for i in range (Xtest.shape[0]):
            #row = [idx2[i], aid[idx2[i]]] + allData.ix[idx2[i],:].tolist() + [Ytest[i], Ypred[i], Yscore[i][1]]
            row = [aid[idx2[i]], Ytest[i], Ypred[i], float("{0:.2f}".format(Yscore[i][1]))]
            csv_writer_pred.writerow(row)
    
    #compute roc
    #y_score = clf.decision_function(Xtest)
    
    y_score = clf.predict_proba(Xtest)
    #print 'y_score first column: ', y_score.shape, y_score
    y_pred = clf.predict(Xtest)
    
    print 'y_pred: ', y_pred   
    y_score = y_score[:,1]
    print 'y_score first column: ', y_score.shape, y_score
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Ytest, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    '''
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.0,1])
    plt.ylim([-0.0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    '''
    
    #print 'FPR', false_positive_rate
    #print 'TPR', true_positive_rate
    
    csv_writer_auc = csv.writer(open(myDir + 'roc.csv', 'wb'), delimiter = ',')
    csv_writer_auc.writerow(false_positive_rate)
    csv_writer_auc.writerow(true_positive_rate)
    
    
    ''' 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    print 'nclass: ', nclass
    for i in range(nclass):
        fpr[i], tpr[i], _ = roc_curve(Ytest[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Ytest.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    '''
    return score, f1Score, cm, Xtest, Ypred, roc_auc 


def mainOrig():     
    #write 'nPosts', 'delRatio','option', 'accuracy', 'F1-score'
    
    csv_writer = csv.writer(open(myDir + market + 'log.csv', 'wb'), delimiter = ',')
    csv_writer.writerow(['nPosts', 'delRatio','option', 'accuracy', 'F1-score'])
    
    #for opt in [1,2,3,4,0]:
    nPostThreshold = 10
    delRatioThreshold = 0.8
    #for delRatioThreshold in [0.7]:
    for nPostThreshold in [5, 10]:
        for delRatioThreshold in [0.5, 0.6]:
        #for delRatioThreshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            #T = 5
            T = (nPostThreshold + 1)/2
            for opt in [0]:
                #featFile = 'feat-balance-' + str(nPostThreshold) + '-' + str(delRatioThreshold) + '.csv'
                featFile = 'feat-balance-' + str(nPostThreshold) + '-' + str(delRatioThreshold) + '-first-' + str(T) + '.csv'
                #featFile = 'feat-balance-10-0.7-first-5.csv'
                score, f1, cm, Xtest, Ypred, roc_auc = classify(myDir + market + featFile, 
                                                    algo = 'ensemble', option = opt, 
                                                    logFile = myDir + market + 'featImp.csv', 
                                                    predFile = None)
                csv_writer.writerow([nPostThreshold, delRatioThreshold, opt, score, f1])
                print 'roc_auc: ', roc_auc   

def main3D():
    #write  nPosts x threshold    
    csv_writer = csv.writer(open(myDir + market + 'log.csv', 'wb'), delimiter = ',')
    csv_writer.writerow(['nPosts x delRatio'])
    csv_writer.writerow([0,0.5, 0.6, 0.7, 0.8, 0.9])
    
    csv_writer_f1 = csv.writer(open(myDir + market + 'log-f1.csv', 'wb'), delimiter = ',')
    csv_writer_f1.writerow(['nPosts x delRatio'])
    csv_writer_f1.writerow([0,0.5, 0.6, 0.7, 0.8, 0.9])
    #for opt in [1,2,3,4,0]:
    
    for nPostThreshold in [5, 10, 15, 20]:
    #for nPostThreshold in [10]:
        scoreList = []     
        f1List = []   
        for delRatioThreshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        #for delRatioThreshold in [0.7]:
            #T = 5
            T = (nPostThreshold + 1)/2
            for opt in [0]:
                featFile = 'feat-balance-' + str(nPostThreshold) + '-' + str(delRatioThreshold) + '.csv'
                #featFile = 'feat-balance-' + str(nPostThreshold) + '-' + str(delRatioThreshold) + '-first-' + str(T) + '.csv'
                
                #featFile = 'feat-balance-10-0.7-first-5.csv'
                
                #featFile = 'feat-balance-' + str(nPostThreshold) + '-' + str(delRatioThreshold) + '-first-' + str(T) +  '-ignorefirst' + str(T) + '.csv' 
                
                score, f1, cm, Xtest, Ypred, roc_auc = classify(myDir + market + featFile, 
                                                    algo = 'ensemble', option = opt, 
                                                    logFile = myDir + market + 'featImp.csv', 
                                                    predFile = None)
                scoreList.append(score)
                
                f1List.append(f1)
                print 'roc_auc: ', roc_auc   
        #write scoreList and f1-score
        csv_writer.writerow([nPostThreshold] + scoreList)
        csv_writer_f1.writerow([nPostThreshold] + f1List)
 
if __name__ == '__main__':
    myDir = '/Users/longtle/Documents/cqa/br2-struggle/answering/'
    market = 'pl-'
    print 'market: ', market
    main3D()
    pass