'''
Created on Dec 22, 2015

@author: longtle
'''
import re, math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import csv
import pandas as pd

from textstat.textstat import textstat
import time

from sklearn.preprocessing import Imputer, label_binarize
from sklearn import svm, preprocessing, cross_validation, linear_model, tree, ensemble, metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.decomposition.tests.test_nmf import random_state
from sklearn.metrics.metrics import confusion_matrix, f1_score, roc_curve, auc

WORD = re.compile(r'\w+')



myDir = "./data/"

def generateFinal(positiveFile, negativeFile, outputFile = None):
    
    positive = pd.read_csv(positiveFile)
    negative = pd.read_csv(negativeFile)
    
    print 'positive: ', positive.shape
    print 'negative: ', negative.shape
    
    n1, _ = positive.shape
    n2, _ = negative.shape
    print 'n1, n2: ', n1, n2
    
    #allData = pd.concat([positive, negative])
    allData = pd.concat([positive, negative], ignore_index=True)
    #allData.set_index([i for i in range (n1 + n2)])
    #print 'allData: ', allData
    
    allData['q_content'].fillna('', inplace = True)
    allData['a_content'].fillna('', inplace = True)
    
    allFeat = allData.loc[:,["answer_id","warns","spam_report_count","warn_ban_time","rank_id","thanks_count","time_to_answered","friends_count", "asker_rank_id",
                             'deg_adj', 'ego', 'CC', 'CC_adj', 'ego_out', 'ego_adj', 
                             'n_answers', 'n_questions']]
    #allFeat = allData
    #extract some feature based on existing columns
    u_grade = extractFeat(allData, 'u_grade_id')
    #print 'u_grade: ', u_grade.shape, u_grade
    
    q_grade = extractFeat(allData, 'q_grade_id')
    #print 'q_grade: ', q_grade.shape, q_grade
    
    length = extractFeat(allData, 'a_content')
    #print 'length: ', length.shape, length
    
    sim = extractFeat(allData, 'sim')
    #print 'sim: ', sim.shape, sim
    
    clientType = extractFeat(allData, 'client_type_id')
    #print 'Client Type: ', clientType.shape, clientType
    
    containTex = extractFeat(allData, 'tex')
    
    formatText =  extractFeat(allData, 'well_format')
    
    ari =  extractFeat(allData, 'ari')
    
    fres = extractFeat(allData, 'fres')
    
    #append new feats to allFeat
    allFeat.loc[:,'length'] = length
    allFeat.loc[:,'u_grade'] = u_grade
    allFeat.loc[:,'q_grade'] = q_grade
    allFeat.loc[:,'sim'] = sim
    allFeat.loc[:,'diff_grade(answerer_question)'] = allFeat.loc[:,'u_grade'] - allFeat.loc[:,'q_grade']
    allFeat.loc[:,'diff_rank(answerer_asker)'] = allFeat.loc[:,'rank_id'] - allFeat.loc[:,'asker_rank_id']
    allFeat.loc[:,'client_type'] = clientType
    allFeat.loc[:,'containTex'] = containTex
    allFeat.loc[:,'well_format'] = formatText
    allFeat.loc[:, 'ari'] = ari
    allFeat.loc[:, 'fres'] = fres
    allFeat.loc[:, 'typing_speed'] = allFeat.loc[:, 'length'] / allFeat.loc[:, 'time_to_answered']
    allFeat.loc[:, 'typing_speed'] = allFeat.loc[:, 'typing_speed'].round()
    #drop some column
    #allFeat = allFeat.drop(["tl_uid","answer_uid","question_id","answer_id","moderator_response","warn_ban_time","q_content","asker_uid","a_content","date_deleted","asker_rank_id"], axis = 1)
    
    #replace NaN value by mean
    #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    #X = imp.fit_transform(X) 
    
    #add class label
    Y = pd.Series(['P' for i in range (n1)] + ['N' for j in range (n2)])
    #print 'Y: ', Y.shape, Y
    allFeat.loc[:,'outcome'] = Y
    #print 'X[0]: ', X[0,:]
    if (outputFile != None):
        allFeat.to_csv(outputFile, index = False)
        #allFeat.to_csv(outputFile, index = True, index_label = 'idx')



def extractFeat(df, header):
    if (header == 'a_content'):
        feat = df[header].str.len()
    elif (header == 'u_grade_id'):
        feat = df.apply(lambda row:labelGrade(row, 'u_grade_id'), axis = 1)
    elif (header == 'q_grade_id'):
        feat = df.apply(lambda row:labelGrade(row, 'q_grade_id'), axis = 1)
    elif (header == 'sim'):
        feat = df.apply(lambda row:labelSimAQ(row), axis = 1)
    elif (header == 'client_type_id'):
        feat = df.apply(lambda row:labelClientType(row), axis = 1)
    elif (header == 'tex'):
        feat = df.apply(lambda row:labelContent(row, ['[tex]']), axis = 1)
    elif (header == 'well_format'):
        feat = df.apply(lambda row:labelContent(row, ['<br/>', '<em>', '<strong>', '<br />']), axis = 1)
    elif (header == 'ari'):
        feat = df.apply(lambda row:labelARI(row), axis = 1)
    elif (header == 'fres'):
        feat = df.apply(lambda row:labelFRES(row), axis = 1)
    return feat

def joinFeat(market = None, ds = None, n = None):
    '''
    Give a feature matrix and index
    we want to add new feature: feat[indexFeat] based on dictValue
    '''
    if (market == None):
        market = 'us'
    if (ds == None):
        ds = 'deleted'
    if (n == None):
        n = '-10k'
    inputFile  = myDir + market + "-answer-feat-" + ds + n +".csv"
    outputFile = myDir + market + "-answer-feat-" + ds + n + "-all.csv"
    
    orig = pd.read_csv(inputFile)    
    
    #join thanks count
    tc = pd.read_csv(myDir + market + "-thanks-count.csv")
    
    new = pd.merge(orig, tc, how = 'left', left_on = 'answer_uid', right_on = 'user_id' )
    #new = pd.merge(left, right, how = 'left', on = 'user_id' )
    
    #join the typing speed
    speed = pd.read_csv(myDir + market + "-answer-time.csv")
    new = pd.merge(new, speed, how = 'left', on = 'answer_id')
    
    #drop us/es from rank_id
    new['rank_id'] = new['rank_id'].str.replace(market, '')
    new['asker_rank_id'] = new['asker_rank_id'].str.replace(market, '')
    
    fc = pd.read_csv(myDir + market + "-friends-count.csv")
    #new = pd.merge(new, fc, how = 'left', left_on = 'answer_id', right_on = 'user_id' )
    new = pd.merge(new, fc, how = 'left', on = 'user_id' )
    
    aqc = pd.read_csv(myDir + market + "-aq-count.csv")
    new = pd.merge(new, aqc, how = 'left', on = 'user_id' )
    
    network = pd.read_csv(myDir + market + "-netsimile.csv")
    #print 'network: ', network
    
    new = pd.merge(new, network, how = 'left', on = 'user_id')
    
    new.to_csv(outputFile)

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
    
def labelClientType(row):
    # mobile: 1 and desktop 2
    #print 'clientType: ', row['client_type_id']
    if (row['client_type_id'] in [0, 1] ):
        return 1
    else:
        return 0
    
def labelSimAQ(row):
    #print 'row: ', row
    sim = get_cosine(text_to_vector(row['q_content']), text_to_vector(row['a_content']))
    #print 'sim: ', sim, row['q_content'], row['a_content']
    return sim

def labelContent(row, myStrList):
    for myStr in myStrList:
        if (myStr in row['a_content']):
            return 1
    return 0

def labelARI(row):
    # return the Automated Readability Index
    if (pd.isnull(row['a_content'])):
        return 
    #print 'a_content: ', row['a_content']
    try:
        ari = textstat.automated_readability_index(row['a_content'])
        return ari
    except:
        return 

def labelFRES(row):
    #return the Flesch Reading Ease Score. Following table is helpful to access the ease of readability in a document.
    if (pd.isnull(row['a_content'])):
        return 
    try:
        fres = textstat.flesch_reading_ease(row['a_content'])
        return fres
    except:
        return 0

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    #print 'text:', text
    
    words = WORD.findall(text)
    #print 'words: ', words
    return Counter(words)
 
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
    
    aid = allData["answer_id"]
    del X["answer_id"]
    print 'X size:', X.shape, allData.shape
    #del X["idx"]
    
    #select the set of features based on option
    if (option ==  None or option == 0):
        print 'Get all features'
    elif (option == 1):
        X = X[['rank_id', 'u_grade', 'n_answers', 'n_questions']]
    elif (option == 2):
        X = X[['warns', 'spam_report_count', 'warn_ban_time', 'thanks_count', 'friends_count', 'deg_adj', 'ego','CC','CC_adj','ego_out','ego_adj']]
    elif (option == 3):
        X = X[['length', 'containTex','well_format','ari', 'fres']]
    elif (option == 4):
        X = X[['asker_rank_id', 'q_grade', 'sim', 'diff_grade(answerer_question)','diff_rank(answerer_asker)','client_type']]
    else:
        print 'Invalid option'
        return
    featList = list(X.columns.values)
    
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
    #lb = preprocessing.LabelBinarizer()
    #Y = lb.fit_transform(Y)
    le = preprocessing.LabelEncoder()
    le.fit(['P', 'N'])
    Y = le.transform(Y)
    
    print 'Original Y: ', Y.shape, Y
    
    indices = np.arange(X.shape[0])
    Xtrain, Xtest, Ytrain, Ytest, _, idx2 = cross_validation.train_test_split(X, Y, indices, test_size=0.3, random_state = 0)
    print Xtrain.shape, Xtest.shape
    print 'Xtrain :', Xtrain.shape#, Xtrain[0,:]
    print 'Ytrain:', Ytrain.shape
    #print 'idx2: ', idx2.shape, idx2
    
    if (algo == 'logreg'):
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
    
    y_score = clf.predict_proba(Xtest)
    #print 'y_score first column: ', y_score.shape, y_score
    y_pred = clf.predict(Xtest)
    
    #print 'y_pred: ', y_pred   
    y_score = y_score[:,1]
    #print 'y_score first column: ', y_score.shape, y_score
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Ytest, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    return score, f1Score, cm, Xtest, Ypred, roc_auc 

if __name__ == '__main__':
    
    csv_writer = csv.writer(open(myDir + 'log.csv', 'wb'), delimiter = ',')
    csv_writer.writerow(['option', 'accuracy', 'F1-score'])
    #for opt in [1,2,3,4,0]:
    for opt in [0]:
        score, f1, cm, Xtest, Ypred, roc_auc = classify(myDir + 'us-answer-feat-100k.csv', algo = 'bagging', option = opt, 
                                               logFile = myDir + 'featImp.csv', 
                                               predFile = myDir + 'us-answer-pred-100k.csv')
        csv_writer.writerow([opt, score, f1])
        print 'roc_auc: ', roc_auc
    pass