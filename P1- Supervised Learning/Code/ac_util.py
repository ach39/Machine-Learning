# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:18:26 2017

@author: ac104q

https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

"""

import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
from time import clock
from sklearn import datasets
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn.metrics import precision_score, recall_score, \
            accuracy_score, f1_score, roc_auc_score, roc_curve, auc,\
            confusion_matrix, classification_report
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import normalize

#from sklearn.grid_search import GridSearchCV - deprecated

#print sklearn.__version__    : 0.19.0


import matplotlib
matplotlib.rcParams.update({'font.size': 13})
#matplotlib.rcParams['figure.titlesize'] = 'large'
matplotlib.rcParams['figure.figsize'] = 4, 3

import warnings
warnings.filterwarnings("ignore")

#define datasets
IRIS_DS =0
ABALONE_DS =1
BANK_DS =0
BEST=1


def feature_imp(feat_names ,trainX, trainY):
    #feat_names = df.columns[:-1]
    forest = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    forest.fit(trainX,trainY)
    importances = forest.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    for f in range(trainX.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
            feat_names[indices[f]],
            importances[indices[f]]))


    plt.title('Feature Importances')
    plt.bar(range(trainX.shape[1]),
        importances[indices],
        color='blue',
        align='center')
    plt.xticks(range(trainX.shape[1]),
               feat_names[indices], rotation=90)
    plt.xlim([-1, trainX.shape[1]])
    plt.tight_layout()

    # select top features
    model = SelectFromModel(forest, prefit=True)
    X_new = model.transform(trainX)
    print X_new[:5,] 

def plot_roc(fpr,tpr,clf_name):
       # Calculate the AUC
        roc_auc = auc(fpr, tpr)
        print 'ROC AUC: %0.4f' % roc_auc
        
        # Plot of a ROC curve for a specific class
        plt.figure()
        title_obj = plt.title(clf_name + ' : ROC ')
        plt.setp(title_obj, color='b')
        plt.plot(fpr, tpr, label='(auc = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title()
        plt.legend(loc="lower right")
        plt.show()
        
def get_f1score(clf,trainX,trainY,testX,testY, clf_name, roc=False, hdr=True):
    
    digit = 5
    st = clock()
    clf.fit(trainX,trainY)
    train_time = round(clock()-st, 5)
    
    train_acc   = round(accuracy_score(trainY, clf.predict(trainX)), digit)
    
    st =clock()
    y_pred   = clf.predict(testX)    
    test_time = round(clock()-st, 5)
    
    test_acc = round(accuracy_score(testY, y_pred ), digit)
    f1       = round(f1_score(testY, y_pred, average='weighted'), digit)
    recall   = round(recall_score(testY, y_pred, average='weighted'), digit)
    prec     = round(precision_score(testY, y_pred, average='weighted'), digit)  
    
    if roc :
        auc_val = round(roc_auc_score(testY, y_pred),digit)
        if hdr : print ('Clf_name \t Train_Acc  \t Test_Acc  \t F1 score \t Recall \t Precision \t Auc \t train-ms \t test-ms ')                                   
        print clf_name , '\t', train_acc ,'\t', test_acc ,'\t', f1,'\t', recall ,'\t', prec, '\t', auc_val ,'\t', train_time ,'\t', test_time 
        
        #fpr, tpr, thresholds = roc_curve(testY, clf.predict_proba(testX)[:,1])
        fpr, tpr, thresholds = roc_curve(testY, y_pred)
       # plot_roc(fpr,tpr,clf_name)
       

    else :  
        if hdr : print ('Clf_name \t Train_Acc  \t Test_Acc  \t F1 score \t Recall \t Precision \t train-ms \t test-ms ')                                  
        print clf_name , '\t', train_acc ,'\t', test_acc ,'\t', f1,'\t\t', recall ,'\t', prec ,'\t', train_time ,'\t', test_time 
    
#    print ("\n -------------------------------------------------------------------\n")
#    print "Classifier -- " , clf_name    
#    print 'Accuracy :', round(accuracy_score(testY, y_pred), digit)                                 
#    print 'F1 score :', round(f1_score(testY, y_pred, average='weighted'), digit)
#    print 'Recall   :', round(recall_score(testY, y_pred, average='weighted'), digit)
#    print 'Precision:', round(precision_score(testY, y_pred, average='weighted'), digit)
   


def compute_score_bank_base():
    X,Y,trainX,testX, trainY, testY = get_bank_train_test()
    clf_knn = KNeighborsClassifier()
    clf_svm = SVC(random_state=100, kernel='linear')  
    clf_dt = tree.DecisionTreeClassifier(random_state=100 )
    clf_bdt = AdaBoostClassifier(random_state=100) 
    clf_ann = MLPClassifier(random_state=100)

    print ('Clf_name \t Train_Acc  \t Test_Acc  \t F1-score \t Recall \t Precision \t train-ms \t test-ms ') 
    get_f1score(clf_knn, trainX, trainY, testX, testY, "KNN-Bank_def", hdr=False)
    get_f1score(clf_svm, trainX, trainY, testX, testY, "SVM-Bank_def", hdr=False)      
    get_f1score(clf_dt,  trainX, trainY, testX, testY, "DT-Bank_def",  hdr=False)                
    get_f1score(clf_bdt, trainX, trainY, testX, testY, "BOOST-Bank_def",hdr=False)     
    get_f1score(clf_ann, trainX, trainY, testX, testY, "ANN-Bank_def", hdr=False)  
    
    
def compute_score_bank():  
    X,Y,trainX,testX, trainY, testY = get_bank_train_test()
  
    clf_knn = KNeighborsClassifier(metric='manhattan',n_neighbors=10, weights='uniform')
    clf_svm = SVC(random_state=100, kernel='rbf', C=2.0, gamma=0.1)  
    clf_dt = tree.DecisionTreeClassifier(random_state=100 ,min_samples_split= 2, splitter= 'random', \
                criterion= 'entropy', max_depth= 10)              
    DTC = tree.DecisionTreeClassifier(random_state = 100,criterion= 'entropy', splitter= 'random',max_depth= 2)   
    clf_bdt = AdaBoostClassifier(random_state=100, base_estimator = DTC, n_estimators= 100, learning_rate= 1)
  
    clf_ann = MLPClassifier(random_state=100, max_iter=2000 ,early_stopping=True ,\
                            alpha= 0.0001, activation='relu', solver='lbfgs', hidden_layer_sizes=(4L,))

    print ('Clf_name \t Train_Acc  \t Test_Acc  \t F1-score \t Recall \t Precision \t train-ms \t test-ms ') 
    get_f1score(clf_knn, trainX, trainY, testX, testY, "KNN-Bank", hdr=False)
    get_f1score(clf_svm, trainX, trainY, testX, testY, "SVM-Bank", hdr=False)      
    get_f1score(clf_dt,  trainX, trainY, testX, testY, "DT-Bank",  hdr=False)                
    get_f1score(clf_bdt, trainX, trainY, testX, testY, "BOOST-Bank",hdr=False)     
    get_f1score(clf_ann, trainX, trainY, testX, testY, "ANN-Bank", hdr=False)   
    
    
def compute_score_abalone():
    
    X,Y,trainX,testX, trainY, testY = get_abalone_train_test()
  
    #clf_knn = KNeighborsClassifier(metric='manhattan',n_neighbors=30, weights='uniform')
    clf_knn = KNeighborsClassifier(metric='euclidean',n_neighbors=40, weights='uniform')
  
    clf_svm = SVC(random_state=100, kernel='rbf', C=2.0, gamma=1.0)
    
    clf_dt = tree.DecisionTreeClassifier(random_state=100 ,min_samples_split= 2, splitter= 'best', \
                criterion= 'entropy', max_depth= 5)
    
    DTC = tree.DecisionTreeClassifier(random_state = 100,criterion= 'entropy', splitter= 'best',max_depth= 1)   
    clf_bdt = AdaBoostClassifier(random_state=100, base_estimator = DTC, n_estimators= 200, learning_rate= 0.5)
    
    #clf_ann = MLPClassifier(random_state=1,learning_rate_init=0.01, max_iter=2000 ,early_stopping=true\
    #                        alpha= 0.001, activation='relu', solver='adam', hidden_layer_sizes=(20L, 20L))
    
    clf_ann = MLPClassifier(random_state=100, max_iter=2000 ,early_stopping=True ,\
                            alpha= 1, activation='logistic', solver='lbfgs', hidden_layer_sizes=(5,))


    print ('Clf_name \t Train_Acc  \t Test_Acc  \t F1-score \t Recall \t Precision \t train-ms \t test-ms ') 
    get_f1score(clf_knn, trainX, trainY, testX, testY, "KNN-Abalone", hdr=False)
    get_f1score(clf_svm, trainX, trainY, testX, testY, "SVM-Abalone", hdr=False)      
    get_f1score(clf_dt,  trainX, trainY, testX, testY, "DT-Abalone",  hdr=False)                
    get_f1score(clf_bdt, trainX, trainY, testX, testY, "BOOST-Abalone",hdr=False)     
    get_f1score(clf_ann, trainX, trainY, testX, testY, "ANN-Abalone", hdr=False)     

    plot_learning_curve(clf_knn, "KNN_abalone : LearningCurve", X,Y).show()
    plot_learning_curve(clf_svm, "SVM_abalone : LearningCurve", X,Y).show()
    plot_learning_curve(clf_dt, "DT_abalone : LearningCurve", X,Y).show()
    plot_learning_curve(clf_bdt, "BOOST_abalone : LearningCurve", X,Y).show()
    plot_learning_curve(clf_ann, "ANN_abalone : LearningCurve", X,Y).show()
  
  

def compute_score_phishing():
    X,Y,trainX,testX, trainY, testY = get_phishing_train_test()
  
    clf_knn = KNeighborsClassifier(metric='manhattan',n_neighbors=6, weights='distance')
  
    clf_svm = SVC(random_state=100, kernel='rbf', C=10, gamma=0.1)
    
    clf_dt = tree.DecisionTreeClassifier(random_state=100, splitter= 'best', \
                criterion= 'entropy', max_depth= 10)
    
    DTC = tree.DecisionTreeClassifier(random_state = 100,criterion= 'entropy', splitter= 'random',max_depth= 5)   
    clf_bdt = AdaBoostClassifier(random_state=100, base_estimator = DTC, n_estimators= 100, learning_rate= 0.01)
        
    clf_ann = MLPClassifier(random_state=100, max_iter=2000 ,early_stopping=True ,\
                            alpha= 0.1, activation='relu', solver='lbfgs', hidden_layer_sizes=(30L,))

    print ('Clf_name \t Train_Acc  \t Test_Acc  \t F1-score \t Recall \t Precision \t AUC \t train-ms \t test-ms ') 
    get_f1score(clf_knn, trainX, trainY, testX, testY, "KNN-phishing", hdr=False,roc=True)
    get_f1score(clf_svm, trainX, trainY, testX, testY, "SVM-phishing", hdr=False,roc=True)      
    get_f1score(clf_dt,  trainX, trainY, testX, testY, "DT-phishing",  hdr=False,roc=True)                
    get_f1score(clf_bdt, trainX, trainY, testX, testY, "BOOST-Phishing",hdr=False,roc=True)    
    get_f1score(clf_ann, trainX, trainY, testX, testY, "ANN-phishing", hdr=False,roc=True)    
  
      
"""
    Generate a simple plot of the test and training learning curve.
"""    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1, 10) , store=True):
    #ylim = (0.85,1.02)
    if cv == None :
        cv = ms.ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
   
    #print (" new LC - training only ")
    trainX,testX, trainY, testY = ms.train_test_split(X,y,test_size = .3,stratify = y,random_state = 0) 
    X,y = trainX,trainY

    plt.figure()
    title_obj = plt.title(title)
    plt.setp(title_obj, color='b')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    train_sizes_abs, train_scores, test_scores = ms.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    train_scores_max = np.max(train_scores, axis=1)
    test_scores_max = np.max(test_scores, axis=1)
    print train_scores_max
    print ""
    print test_scores_max
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.15,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.15, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Train")
    plt.plot(train_sizes, test_scores_mean, 'o--', color="g",
             label="Test")

    plt.legend(loc="best")
    

    
    if store : 
           out = {'train_sz_abs' : train_sizes_abs,
           'train_sz' : train_sizes,
           'train_mean' : train_scores_mean,
           'train_std' : train_scores_std ,
           'test_mean' : test_scores_mean,
           'test_std' : test_scores_std 
           }
           out = pd.DataFrame(out) 
           fname = title.split(':')[0].strip()
           out['clf'] = fname
           out.to_csv('output/{}_LC_score.csv'.format(fname))
        
    return plt
 
def plot_it(df,x_param , title="") :
   
    x = df[x_param]              

    #df['train_lower'] = df['mean_train_score'] + df['std_train_score']
    #df['train_upper'] = df['mean_train_score'] - df['std_train_score']
    
    plt.plot(x, df['mean_train_score'] , 'r--' , label = "train")
    #plt.fill_between(x, df['train_lower'] , df['train_upper'], color='y', alpha=0.8)
    
    plt.plot(x, df['mean_test_score'] , 'c--' , label = "test")
    
    title_obj = plt.title(title)
    plt.setp(title_obj, color='b')
    plt.legend(loc='best')
    plt.xlabel(x_param)
    plt.ylabel('Accuracy')
    plt.grid(True)
    return plt
    
    
def save_gs_results(gs, clfname="") :
    res_df = pd.DataFrame.from_dict(gs.cv_results_)
    
    writer = pd.ExcelWriter('output/'+ clfname +'.xlsx', engine='xlsxwriter')
    res_df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    return res_df


def plot_validation_time(df, title, ylim=(-0.001 , 0.005)) :
    x= range(df.shape[0])        
    plt.scatter(x, df['mean_fit_time']  ,c='m' ,s=12 ,label = "mean_fit_time")    
    plt.scatter(x, df['mean_score_time'],c='c' ,s=12 ,label = "mean_score_time")
    
    title_obj = plt.title(title)
    plt.setp(title_obj, color='b')
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('time')
    plt.ylim(ylim)
    plt.grid(True)
    return plt
   
def plot_learning_time(X, Y, clf, title="", ylim=(-0.001 , 0.05), store=True):

    out = defaultdict(dict)
    sz = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for frac in sz:    
        X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=1-frac, random_state=0)
        st = clock()
        np.random.seed(100)
        clf.fit(X_train,y_train)
        out['train'][frac]= clock()-st
        st = clock()
        clf.predict(X_test)
        #out['test'][frac]= clock()-st
        out['test'][round(1-frac,1)]= clock()-st
                             
    out = pd.DataFrame(out)
    x= range(out.shape[0])            
    plt.scatter(sz, out['train'] , c='m' , label = "train")    
    plt.scatter(sz, out['test'], c='c' , label = "test")
    title_obj = plt.title(title)
    plt.setp(title_obj, color='b')
    plt.legend()
    plt.xlabel('train size')
    plt.ylabel('time (ms)')
    plt.ylim(ylim)
    plt.grid(True)
   
    if store : 
        fname = title.split(':')[0].strip()
        out.to_csv('output/{}_timing.csv'.format(fname))
    
    return plt
  
                                           
'''
 Tune params. 
'''
def tuneParams(clf,trainX, trainY, testX, testY, params ,clfname="" ,verbose=False):
    np.random.seed(10)
    gs = ms.GridSearchCV(clf,              
            param_grid =params,
            scoring ='accuracy',
            cv =5,
            refit =True,
            verbose=0)
             
    gs.fit(trainX,trainY)
    print(gs.best_score_)
    print(gs.best_params_)
  
    if verbose : 
        for x in gs.grid_scores_ : print x

    res_df = save_gs_results(gs , clfname)
    
    # test on best estimator
    best_clf = gs.best_estimator_
    best_clf.fit(trainX, trainY)
    
    print clfname
    print('Train accuracy: %.4f' % best_clf.score(trainX, trainY))
    print('Test accuracy: %.4f' % best_clf.score(testX, testY))

    return best_clf , res_df   
    
##############################################################################    
#    
# read Datasets
#    
#############################################################################
def get_iris_train_test(test_size=0.2, shuffle=False ):
    #f = "C:/Users/ac104q/Desktop/MY_DATA/Training/Transformation 2020/3.OMCS/2018- Spring/ML/datasets/iris.csv"
    f = "datasets/iris.csv"
    df = pd.read_csv(f)
    
    if (shuffle) : df = df.sample(frac=1)  ## shuffle the data ???
    
    # Change string value to numeric
    df.set_value(df['species']=='Iris-setosa',['species'],0)
    df.set_value(df['species']=='Iris-versicolor',['species'],1)
    df.set_value(df['species']=='Iris-virginica',['species'],2)
    df = df.apply(pd.to_numeric)
    
    df_array = df.as_matrix()
    X = df_array[:,:4]
    Y = df_array[:,4]
    
    trainX,testX, trainY, testY = ms.train_test_split(df_array[:,:4],
                                                      df_array[:,4],
                                                      test_size = test_size,
                                                      #stratify = Y,
                                                      random_state = 0)
                                                     
    sc = StandardScaler()
    ##sc.fit(trainX) ; trainX = sc.transform(trainX)
    trainX= sc.fit_transform(trainX)
    testX = sc.transform(testX)
    
    #feature_imp(df.columns[:-1],trainX, trainY)
    
    print('TrainX :', trainX.shape , " |  trainY : ", trainY.shape )
    print('TestX :', testX.shape ,   " |  TestY : ", testY.shape )
    
    return X,Y,trainX,testX,trainY,testY
     

#test_size,scale, shuffle ,drop_gender = 0.3,True, False, False 

def get_abalone_train_test( test_size=0.3, scale = True, shuffle=False ,drop_gender = False ):

    f = "datasets/abalone_ac.csv"
    col_names = ["sex", "len", "diameter", "ht", "whole_wt", "shuck_wt", "visc_wt", "shell_wt", "rings"]
    df = pd.read_csv(f, names=col_names)
    
    print "Reading  " , f
    print("Number of samples: %d" % len(df))
    df.head()
    df.rings.unique()

    #if (shuffle) : df = df.sample(frac=1)  ## shuffle the data ???

    #dummify gender
    #for label in "MFI": df[label] = df["sex"] == label
    df_sex = pd.get_dummies(df['sex'])
    df = df.join(df_sex)
         
   # Change string value to numeric
    df.set_value(df['rings']=='Small',['rings'],0)
    df.set_value(df['rings']=='Medium',['rings'],1)
    df.set_value(df['rings']=='Large',['rings'],2)
    
    if drop_gender :
        df = df.drop(['M', 'F', 'I'], axis=1)
 
    del df['sex']
    df = df.apply(pd.to_numeric)
    
    Y = df.rings.values
    del df['rings'] 
    X = df.values.astype(np.float)
    
    trainX,testX, trainY, testY = ms.train_test_split(X,Y,
                                                      test_size = test_size,
                                                      stratify = Y,
                                                      random_state = 0)
                                                      
    ## gender shouldn't be scaled. 
    if scale :                                                 
        sc = StandardScaler()
        trainX= sc.fit_transform(trainX)
        testX = sc.transform(testX)
        print "sc mean  : " , sc.mean_ 
        print "sc_scale : " , sc.scale_
        print "sc_var   : " , sc.var_
    
    #feature_imp(df.columns[:],trainX, trainY)
    
    print('TrainX :', trainX.shape , " |  trainY : ", trainY.shape )
    print('TestX :', testX.shape ,   " |  TestY : ", testY.shape )
    
    return X,Y,trainX,testX,trainY,testY


#test_size, scale, read_shuf =0.2 , True , True 
def get_bank_train_test(test_size=0.2, scale = True, read_shuf = True ):

    if read_shuf :
        f = "datasets/banknote_auth_shuffled.csv"
        print "reading " , f
        df = pd.read_csv(f, index_col=0) 
    else :
        f = "datasets/banknote_authentication.csv"
        col_names = ["var", "skew", "curtosis", "entropy", "label"]
        df = pd.read_csv(f,names = col_names)
        #df = shuffle(df)
        #df.to_csv("datasets/banknote_auth_shuf_1.csv")
 
    print("Number of samples: %d" % len(df))
    df.head()
    df.label.unique()

    df_array = df.as_matrix()
    X = df_array[:,:4]
    Y = df_array[:,-1]
        
    trainX,testX, trainY, testY = ms.train_test_split(X,Y,
                                                      test_size = test_size,
                                                      stratify = Y,
                                                      random_state = 0)     
    if scale :                                                 
        sc = StandardScaler()
        trainX= sc.fit_transform(trainX)
        testX = sc.transform(testX)
        print "sc mean  : " , sc.mean_ 
        print "sc_scale : " , sc.scale_
        print "sc_var   : " , sc.var_
    
    #feature_imp(df.columns[:-1],trainX, trainY)
    
    print('TrainX :', trainX.shape , " |  trainY : ", trainY.shape )
    print('TestX :', testX.shape ,   " |  TestY : ", testY.shape )
    
    return X,Y,trainX,testX,trainY,testY


def get_phishing_train_test(test_size=0.3 ):
    f = "datasets/phishing_ac.csv"
    print "Reading  " , f
    df = pd.read_csv(f) 

 
    print("Number of samples: %d" % len(df))
    df.head()
    #print df.label.unique()

    df_array = df.as_matrix()
    X = df_array[:,:30]
    Y = df_array[:,-1]
        
    trainX,testX, trainY, testY = ms.train_test_split(X,Y,
                                                      test_size = test_size,
                                                      stratify = Y,
                                                      random_state = 0)     
    
    #feature_imp(df.columns[:-1],trainX, trainY)
    
    print('TrainX :', trainX.shape , " |  trainY : ", trainY.shape )
    print('TestX :', testX.shape ,   " |  TestY : ", testY.shape )
    
    return X,Y,trainX,testX,trainY,testY





def my_test() :
    
    nbr = 40
    X,Y,trainX,testX, trainY, testY = get_abalone_train_test(scale=True,drop_gender=False)
  
    clf_knn = KNeighborsClassifier(metric='manhattan',n_neighbors=nbr, weights='uniform')
    
    print ('Clf_name \t Train_Acc  \t Test_Acc  \t F1 score \t Recall \t Precision') 
    get_f1score(clf_knn, trainX, trainY, testX, testY, "KNN-Abalone", hdr=False)
    

    X,Y,trainX,testX, trainY, testY = get_abalone_train_test(scale=True,drop_gender=True)
  
    clf_knn = KNeighborsClassifier(metric='manhattan',n_neighbors=nbr, weights='uniform')
    get_f1score(clf_knn, trainX, trainY, testX, testY, "KNN-Abalone", hdr=False)


    X,Y,trainX,testX, trainY, testY = get_abalone_train_test(scale=False,drop_gender=False)
  
    clf_knn = KNeighborsClassifier(metric='manhattan',n_neighbors=nbr, weights='uniform')
    get_f1score(clf_knn, trainX, trainY, testX, testY, "KNN-Abalone", hdr=False)
    
    X,Y,trainX,testX, trainY, testY = get_abalone_train_test(scale=False,drop_gender=True)
  
    clf_knn = KNeighborsClassifier(metric='manhattan',n_neighbors=nbr, weights='uniform')
    get_f1score(clf_knn, trainX, trainY, testX, testY, "KNN-Abalone", hdr=False)








def check_pca(): 
    from sklearn.decomposition import PCA
    X,Y,trainX,testX, trainY, testY = get_phishing_train_test()
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    
    #The amount of variance that each PC explains
    var= pca.explained_variance_ratio_
    
    #Cumulative Variance explains
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    
    #Looking at above plot I'm taking 30 variables
    pca = PCA(n_components=20).fit(X)
    print pca.explained_variance_ratio_
    X1=pca.fit_transform(X)
    
    #pca = PCA(n_components=2)
    #X_r = pca.fit(X).transform(X)



#test_size=0.3
#def get_abalone_train_test_org(test_size=0.3, shuffle=False ):
#
#    f = "datasets/abalone.csv"
#    
#    col_names = ["sex", "len", "diameter", "ht", "whole_wt", 
#                "shuck_wt", "visc_wt", "shell_wt", "rings"]
#    df = pd.read_csv(f, names=col_names)
#    
#    print("Number of samples: %d" % len(df))
#    df.head()
#    df.rings.unique()
#
#    #if (shuffle) : df = df.sample(frac=1)  ## shuffle the data ???
#    
#    #dummify gender
#    df_sex = pd.get_dummies(df['sex'])
#    df = df.join(df_sex)
#        
#    #group labels - >=0 :7 & >=12 : 5
#    df['label'] = 0
#    df.label[df.rings == 9]  = 1
#    df.label[df.rings == 10]  = 1
#    df.label[df.rings >= 11] = 2
#    
#    df = df.drop(['M', 'F', 'I'], axis=1)
#    
#    df.to_csv("abalone_processed_1.csv")
#    Y = df.label.values
#    del df['sex']
#    del df['rings'] 
#    del df['label'] 
#    X = df.values.astype(np.float)
#    df.to_csv("abalone_processed_2.csv")

#    trainX,testX, trainY, testY = ms.train_test_split(X,Y,                            
#                                                      test_size = test_size,
#                                                      stratify = Y,
#                                                      random_state = 0)                                                    
#    ## avoid scaling dummy var- gender
#    sc = StandardScaler()
#    trainX= sc.fit_transform(trainX)
#    testX = sc.transform(testX)
#
#    #feature_imp(df.columns[:],trainX, trainY)
#    
#    df_tmp = pd.DataFrame(trainX)
#    
#    #feature_imp(df.columns[:-1],trainX, trainY)
#    
#    print('TrainX :', trainX.shape , " |  trainY : ", trainY.shape )
#    print('TestX :', testX.shape ,   " |  TestY : ", testY.shape )
#    
#    return X,Y,trainX,testX,trainY,testY


#    out['test'][0.7] = 0.057745819014908284
#    out['test'][0.8] = 0.058845819014908284
#    out['test'][0.9] = 0.061945819014908284


'''

Clf_name         Train_Acc      Test_Acc        F1 score        Recall          Precision
KNN-Abalone     0.72802         0.73365         0.6837          0.73365         0.68057
SVM-Abalone     0.74307         0.73126         0.67612         0.73126         0.67086
DT-Abalone      0.71091         0.71451         0.69777         0.71451         0.68869
BOOST-Abalone   0.7506  	   0.72727       0.68052         0.72727         0.68088
ANN-Abalone     0.73384         0.72807         0.66906         0.72807         0.66595
'''

'''
Impact of scaling and gender on KNN
With scaling and gender included gives best results.

KNN_Abalone        Train_Acc       Test_Acc        F1 score        Recall          Precision
Scl=T,dropGdr=F    0.73315          0.74242         0.69314         0.74242         0.70422
Scl=T,dropGdr=T    0.72802          0.73365         0.6837          0.73365         0.68057
Scl=F,dropGdr=T    0.7287  		  0.72967         0.67892         0.72967         0.6735
Scl=F,dropGdr=F    0.72494          0.73764         0.69524         0.73764         0.69786

'''