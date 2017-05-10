'''
Created on Feb 01, 2017

@author: longtle
'''
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from textstat.textstat import textstat
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer, label_binarize
from sklearn import svm, preprocessing, cross_validation, linear_model, tree, ensemble, metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.decomposition.tests.test_nmf import random_state
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

#myDir = "./data/"
myDir = '/Users/longtle/OneDrive - Rutgers University/cqa/difficulty/'

def labelARI(row):
    # return the Automated Readability Index
    if (pd.isnull(row['question_content'])):
        return 
    #print 'a_content: ', row['a_content']
    try:
        ari = textstat.automated_readability_index(row['question_content'])
        return ari
    except:
        return 
  
def labelGrade(row, header):
    #print 'row: ', row
    if (row[header] == 'us1' or row[header] == 'pl1'):
        return 1
    elif (row[header] == 'us2' or row[header] == 'pl2'):
        return 2
    elif (row[header] == 'us5' or row[header] == 'pl3'):
        return 3
    else:
        #print 'unknown: ', row[header]
        return 0
      
def extractFeat(df, header):
    if (header == 'question_content'):
        feat = df[header].str.len()
    elif (header == 'grade_id'):
        feat = df.apply(lambda row:labelGrade(row, 'grade_id'), axis = 1)
    elif (header == 'q_grade_id'):
        feat = df.apply(lambda row:labelGrade(row, 'q_grade_id'), axis = 1)
    elif (header == 'ari'):
        feat = df.apply(lambda row:labelARI(row), axis = 1)
    elif (header == 'fres'):
        feat = df.apply(lambda row:labelFRES(row), axis = 1)
    return feat

    
def joinFeat(market = None, n = None):
    '''
    Give a feature matrix and index
    we want to add df feature: feat[indexFeat] based on dictValue
    '''
    if (market == None):
        market = 'us'
    if (n == None):
        #n = '-1k'
        n = ''
        
    inputFile  = myDir + market + "-questions" + n +".csv"
    outputFile = myDir + market + "-questions-feat" + n + ".csv"
    
    orig = pd.read_csv(inputFile)    
    
    #join askers and answerers
    users = pd.read_csv(myDir + market + "-users.csv")    
    df = pd.merge(orig, users, how = 'left', left_on = 'asker_id', right_on = 'user_id' )
    
    df.loc[:, 'age_when_posting'] = df['year_posted'] - df['birth_year']
    
    u_grade = extractFeat(df, 'grade_id')
    
    q_grade = extractFeat(df, 'q_grade_id')
    #print 'u_grade: ', u_grade.shape, u_grade
    
    
    length = extractFeat(df, 'question_content')
    #print 'length: ', length.shape, length
    
    ari =  extractFeat(df, 'ari')
    
    df.loc[:,'length'] = length
    df.loc[:,'u_grade'] = u_grade
    df.loc[:,'q_grade'] = q_grade
    df.loc[:, 'ari'] = ari
    
    answerers = users[['user_id', 'rank_id']]
    answerers = answerers.rename(columns={'user_id': 'answerer_id', 'rank_id': 'answerer_rank_id'})
    df = pd.merge(df, answerers, how = 'left', on = 'answerer_id' )
    
    #join draft time
    drafts = pd.read_csv(myDir + market + "-answer-drafts.csv")    
    del drafts['user_id']
    del drafts['question_id']
    
    df = pd.merge(df, drafts, how = 'left', on = 'answer_id') 
    
    df['rank_id'] = df['rank_id'].str.replace(market, '')
    df['answerer_rank_id'] = df['answerer_rank_id'].str.replace(market, '')

    #drop some features

    del df['asker_id']
    del df['q_grade_id']
    del df['q_date_created']
    del df['year_posted']
    del df['answer_id']
    del df['answerer_id']
    del df['answer_content']
    del df['a_date_created']  
    del df['user_id']
    del df['grade_id']
    del df['birth_year']
    
      
    df.to_csv(outputFile, index=False)
    

def findSimilarClustering(listU, nCluster):
    '''
    Grouping listU by clustering (k-means)
    This is unsupervised approach, but we need to select number of cluster
    '''
    y_pred = KMeans(n_clusters= nCluster,random_state=None).fit_predict(listU)
    
    
    
def findSimilarClustering(inputFile, nCluster = 100, outputFile = None):

    df = pd.read_csv(inputFile)
    #df = df.head(1000)
    df = df.fillna(0)
    
    listId = df['question_id'].values.tolist()
    
    listFeat = df.drop(['question_id', 'subject_id', 'question_content', 'best_answer'], 1)
    
    print 'list of header: ', list(listFeat.columns.values)
    
    listFeat = listFeat.as_matrix()
    print 'listFeat shape: ', listFeat.shape
    y_pred = KMeans(n_clusters= nCluster,random_state=None).fit_predict(listFeat)
    if (outputFile != None):
        csv_writer = csv.writer(open(outputFile, 'wb'), delimiter =',')    
        csv_writer.writerow(['user_id', 'cluster_id'])
        for i in range (len(listId)):
            csv_writer.writerow([listId[i], y_pred[i]])
            
def regression(inputFile, algo, outputFile = None):
    #
    allData = pd.read_csv(inputFile)
    print 'allData size: ', allData.shape#, allData.ix[0,:]
    
    allData = allData[np.isfinite(allData['outcome'])]
    allData = allData.reset_index()
    
    Y = allData["outcome"]
    
    
    
    print 'check invalid value: ', np.isnan(Y).any()

    X = allData
    #delete the outcome from predictor X
    del X["outcome"]
    aid = X["question_id"]
    del X["question_id"]
    del X["subject_id"]
    del X["best_answer"]
    print 'X size:', X.shape, allData.shape, X
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    X = imp.fit_transform(X) 
    print 'X shape: ', X.shape #, X
    
    
    #normalize data
    X = preprocessing.scale(X)
    print 'X[0]: ', X[0]
    
    indices = np.arange(X.shape[0])
    Xtrain, Xtest, Ytrain, Ytest, _, idx2 = cross_validation.train_test_split(X, Y, indices, test_size=0.3, random_state = 0)
    print Xtrain.shape, Xtest.shape
    print 'Xtrain :', Xtrain.shape#, Xtrain[0,:]
    print 'Ytrain:', Ytrain.shape
    
    if (algo == 'reg'):
        regr = linear_model.LinearRegression()    
    elif (algo == 'dt'):
        regr= tree.DecisionTreeRegressor(max_depth=3)
    else:
        regr = None
        return
    regr.fit(Xtrain, Ytrain)
    Ypred = regr.predict(Xtest)
    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((regr.predict(Xtest) - Ytest) ** 2))
    # Explained variance score: 1 is perfect prediction
    print("Mean abs error: %.2f", np.mean(np.absolute(regr.predict(Xtest) - Ytest)))
    
    print('Variance score: %.2f' % regr.score(Xtest, Ytest))
        
        
        
    # Plot outputs
    #print 'shape: ', Ytest.shape
    
    idx = [i for i in range (Ytest.shape[0])]
    #print 'idx: ', idx
    plt.scatter(idx, Ytest,  color='black')
    plt.scatter(idx, Ypred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    
    Ytest = Ytest.get_values().tolist()
    #print 'YTest:', Ytest.shape, type(Ytest), Ytest[0] #, Ypred
    print 'YPred:', Ypred.shape, type(Ypred), Ypred[0] #, Ypred
    if (outputFile != None):
        csv_writer_pred = csv.writer(open(outputFile, 'wb'), delimiter = ',')
        #csv_writer_pred.writerow(['idx', 'answer_id'] +  featList + ['outcome', 'prediction', 'score'])
        csv_writer_pred.writerow(['answer_id','outcome', 'prediction'])
        for i in range (Xtest.shape[0]):
            #row = [idx2[i], aid[idx2[i]]] + allData.ix[idx2[i],:].tolist() + [Ytest[i], Ypred[i], Yscore[i][1]]
            print 'i', i, Ytest[i], Ypred[i]
            row = [aid[idx2[i]], Ytest[i], Ypred[i]]
            #row = [Ytest[i], Ypred[i]]
            csv_writer_pred.writerow(row)

if __name__ == '__main__':
    #joinFeat(market = 'us')
    #findSimilarClustering(inputFile = myDir + 'us-questions-feat.csv', nCluster = 5, outputFile = myDir + 'us-question-cluster.csv')
    #for subject in ['biology']:
    for subject in ['physics']:
        regression(myDir + subject + '-final.csv', algo = 'dt', outputFile = myDir + subject + '-pred.csv')
    pass