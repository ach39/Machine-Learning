# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 09:38:57 2017

@author: ac104q
"""

from ac_util import *

##################################################
# Boosting : Validation curve
##################################################   
def plot_boost_gs(res_df) :
    df = res_df[(res_df.param_metric=='') & (res_df.param_weights=='')]   
    plot_it(df , " ").show()

##################################################
# Adaboost - Iris
##################################################    
def run_adaboost_iris():
    X,Y,trainX,testX, trainY, testY = get_iris_train_test()        
#    params = {"base_estimator__criterion" : ["gini", "entropy"],
#              "base_estimator__splitter" :   ["best", "random"],
#              "base_estimator__max_depth" : [1,2,3,4,5,6,7,8,9,10],
#              "n_estimators": [1,2,5,10,20,30,45,60,80,100,500,1000],
#             }
    params = {"base_estimator__criterion" : ["entropy"],
              "base_estimator__splitter"  : ["random"],
              "base_estimator__max_depth" : [1,2,5,7,10],
              "n_estimators" : [5,10,20,30,45,100],
              "learning_rate": [0.01, 0.1, 1, 1.5,2]
             }       
    DTC = tree.DecisionTreeClassifier(random_state = 100)   
    clf = AdaBoostClassifier(base_estimator = DTC)                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "AdaBoost_iris", False)                                                    
    
    print('run_boost : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_boost : Test accuracy: %.3f' % best_clf.score(testX, testY))
        
    #plot validation and learning curve score
    #plot_boost_gs(res_df)
    cv = ms.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(best_clf, "BOOST_iris : Learning Curve", X,Y ,cv=cv).show()
    
    #plot validation and learning curve time
    ylim=(-0.05 , 0.07)
    
    #plot_validation_time(res_df ,"BOOST_Iris : fit/score time",ylim).show()
    plot_learning_time(X,Y,best_clf ,"BOOST_Iris : Time complexity",ylim).show()
    
    
##################################################
# Boosting - abalone
##################################################
def run_adaboost_abalone():

    X,Y,trainX,testX, trainY, testY = get_abalone_train_test()
        
    params = {"base_estimator__criterion" : ["entropy","gini"],
              "base_estimator__splitter"  : ["random", "best"],
              "base_estimator__max_depth" : [1,2,5,7],   
              "n_estimators" : [5,10,50,100,200,600],
              "learning_rate": [0.01, 0.1, 1,2]
             }
   
#    params = {"base_estimator__criterion" : ["entropy"],
#              "base_estimator__splitter"  : ["best"],
#              "base_estimator__max_depth" : [1,2,5,10,30],   
#              "n_estimators" : [10,100,200,500,600],
#              "learning_rate": [ 0.1, 0.5] 
#              }
#  
    DTC = tree.DecisionTreeClassifier(random_state = 100,criterion= 'entropy', splitter= 'random',max_depth= 10)   
    clf_bdt = AdaBoostClassifier(random_state=100, base_estimator = DTC, n_estimators= 600, learning_rate= 0.1)
    
    if(BEST) : 
        params = {"base_estimator__criterion" : ["entropy"],
              "base_estimator__splitter"  : ["best"],
              "base_estimator__max_depth" : [1],   
              "n_estimators" : [200], 
              "learning_rate": [0.5] 
              }
               
    DTC = tree.DecisionTreeClassifier(random_state = 100)   
    clf = AdaBoostClassifier(base_estimator = DTC)                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "AdaBoost_abalone", False)                                                    
        
    print('run_boost : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_boost : Test accuracy: %.3f' % best_clf.score(testX, testY))
        
    cv = ms.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(best_clf, "BOOST_Abalone : Learning Curve", X,Y ,cv=cv).show()    
    plot_learning_time(X,Y,best_clf ,"BOOST_Abalone : Time Complexity",ylim=(-0.05 , 0.9)).show()    
    get_f1score(best_clf,trainX,trainY,testX,testY, "BOOST-Abalone",False)


    
    
##################################################
# Adaboost - Banknote Authentication
##################################################
def run_adaboost_bank():
    X,Y,trainX,testX, trainY, testY = get_bank_train_test()        
    params = {"base_estimator__criterion" : ["entropy"],
              "base_estimator__splitter"  : ["random",'best'],
              "base_estimator__max_depth" : [1,2,5,7,10],   
              "n_estimators" : [5,10,50,100,200,600],
              "learning_rate": [0.01, 0.1, 1, 2]
             }
    params = {"base_estimator__criterion" : ["entropy"],
              "base_estimator__splitter"  : ["random"],
              "base_estimator__max_depth" : [2],   
              "n_estimators" : [100],
              "learning_rate": [1] 
              }
               
    DTC = tree.DecisionTreeClassifier(random_state = 100)   
    clf = AdaBoostClassifier(base_estimator = DTC)                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "AdaBoost_Bank", False)                                                    
        
    print('run_boost : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_boost : Test accuracy: %.3f' % best_clf.score(testX, testY))
        

    #plot_boost_gs(res_df)
    cv = ms.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(best_clf, "BOOST_Bank : Learning Curve", X,Y ,cv=cv).show()
    
    #plot validation and learning curve time
    ylim=(-0.05 , 0.3)
    
    #plot_validation_time(res_df ,"BOOST_Iris : fit/score time",ylim).show()
    plot_learning_time(X,Y,best_clf ,"BOOST_Bank : Time Complexity",ylim).show()
    

##################################################
# Adaboost - Phishing 
##################################################
def run_adaboost_phishing():
    X,Y,trainX,testX, trainY, testY = get_phishing_train_test()        
    params = {"base_estimator__criterion" : ["entropy"],
              "base_estimator__splitter"  : ["random",'best'],
              "base_estimator__max_depth" : [1,2,5,7,10],   
              "n_estimators" : [5,10,50,100,200,600],
              "learning_rate": [0.01, 0.1, 1, 2]
             }
    
    if(BEST) : 
             params = {"base_estimator__criterion" : ["entropy"],
              "base_estimator__splitter"  : ["random"],
              "base_estimator__max_depth" : [5],   
              "n_estimators" : [100],
              "learning_rate": [.01] 
              }
               
    DTC = tree.DecisionTreeClassifier(random_state = 100)   
    clf = AdaBoostClassifier(base_estimator = DTC)                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "AdaBoost_phishing", False)                                                    
        
    print('run_boost : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_boost : Test accuracy: %.3f' % best_clf.score(testX, testY))
        
    cv = ms.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(best_clf, "BOOST_phishing : Learning Curve", X,Y ,cv=cv).show() 
    plot_learning_time(X,Y,best_clf ,"BOOST_phishing : Time Complexity",ylim =(-0.05,3)).show() 
    get_f1score(best_clf,trainX,trainY,testX,testY, "BOOST_phishing",True)

##################################################
# Adaboost - MAIN
##################################################
if __name__== "__main__" :
    print "Adaboost main ... "
    
#    run_adaboost_bank()
#    run_adaboost_abalone()
#    run_adaboost_phishing()










































# Abalone
# Entropy/Random
    #{'n_estimators': 500, 'base_estimator__criterion': 'entropy', 'base_estimator__max_depth': 2, 'learning_rate': 0.1, 'base_estimator__splitter': 'random'}
    #Train accuracy: 0.748
    #Test accuracy: 0.730

# Entropy/best
    #{'n_estimators': 200, 'base_estimator__criterion': 'entropy', 'base_estimator__max_depth': 1, 'learning_rate': 0.5, 'base_estimator__splitter': 'best'}
    #Train accuracy: 0.729
    #Test accuracy: 0.737


#AdaBoost_Bank
#{'n_estimators': 100, 'base_estimator__criterion': 'entropy', 'base_estimator__max_depth': 1, 'learning_rate': 1, 'base_estimator__splitter': 'random'}
#Train accuracy: 1.0000
#Test accuracy: 1.0000



# Iris
    # best
    #{'n_estimators': 5, 'base_estimator__criterion': 'entropy', 
    #'base_estimator__max_depth': 7, 'base_estimator__splitter': 'best'}
    
    #{'n_estimators': 100, 'base_estimator__criterion': 'entropy', 
    #'base_estimator__max_depth': 1, 'learning_rate': 1.5, 'base_estimator__splitter': 'best'}
    #Train accuracy: 0.943
    #Test accuracy: .90
    
    #best so far
    #{'n_estimators': 5, 'base_estimator__criterion': 'entropy', 'base_estimator__max_depth': 5, 
    #'learning_rate': 0.01, 'base_estimator__splitter': 'random'}
    #Train accuracy: 0.983
    #Test accuracy: 1.000
    