# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 09:38:17 2017
@author: ac104q
"""

from ac_util import *




##################################################
# Decision Tree : Validation curve
##################################################   
def plot_dt_gs(res_df, dataset="") :

    #param_criterion	param_max_depth	param_splitter
    
    #res_df = res_df[res_df.param_min_samples_split == 2 ]
    
    df = res_df[(res_df.param_criterion=='entropy') & (res_df.param_splitter=='best')]   
    plot_it(df, 'param_max_depth', dataset+'Entropy/Best' ).show()

    df = res_df[(res_df.param_criterion=='entropy') & (res_df.param_splitter=='random')]   
    plot_it(df, 'param_max_depth', dataset+'Entropy/Random' ).show()
    
    df = res_df[(res_df.param_criterion=='gini') & (res_df.param_splitter=='best')]   
    plot_it(df, 'param_max_depth', dataset+'Gini/Best' ).show()

    df = res_df[(res_df.param_criterion=='gini') & (res_df.param_splitter=='random')]   
    plot_it(df, 'param_max_depth', dataset+'Gini/Random' ).show()
    
##################################################
# SVM - Iris
##################################################    
def run_dt_iris():
    X,Y,trainX,testX, trainY, testY = get_iris_train_test()       
    depth = range(1,11) + [15,20 ,30, None]
    params = { 'criterion' : ['entropy','gini'],
               'max_depth' : depth ,
               'min_samples_split' : [2,4,6,10],
               'splitter' : ["best", "random"] }
      
    
    clf = tree.DecisionTreeClassifier(random_state=100)                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "DT_iris" ,False)  
    print('run_dt : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_dt : Test accuracy: %.3f' % best_clf.score(testX, testY))
                                                  
    #plot validation and learning curve score
    plot_dt_gs(res_df, 'Iris -')
    plot_learning_curve(best_clf, "DT_Iris : LC", X,Y ).show()
      
    #plot validation and learning curve time
    plot_validation_time(res_df ,"DT_Iris : fit/score time").show()
    plot_learning_time(X,Y,best_clf ,"DT_Iris : LearningCurve time").show()
    
##################################################
# Decision Tree- Abalone
##################################################
def run_dt_abalone():
    X,Y,trainX,testX, trainY, testY = get_abalone_train_test()            
    depth = range(1,11) + [15,20 ,30, None]
    params = { 'criterion' : ['entropy','gini'],
               'max_depth' : depth ,
               'min_samples_split' : [2,4,6,10],
               'splitter' : ["best", "random"] }
    if(BEST):
        params = { 'criterion' : ['entropy'],'max_depth' : [5] ,'splitter' : ["best"] }
                               
    clf = tree.DecisionTreeClassifier(random_state=100)                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "DT_Abalone" ,False)  
    print('run_dt : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_dt : Test accuracy: %.3f' % best_clf.score(testX, testY))
                                                  
    # plot_dt_gs(res_df, 'Abalone -')
    
    plot_learning_curve(best_clf, "DT_Abalone : Learning Curve", X,Y ).show() 
    #plot_learning_curve(best_clf, "DT_Abalone : Learning Curve-2", trainX,trainY ).show() 
    plot_learning_time(X,Y,best_clf ,"DT_Abalone :Time Complexity", ylim=(-0.01, 0.04)).show()
    
    get_f1score(best_clf,trainX,trainY,testX,testY, "SVM_Abalone",False,True)
    
##################################################
# Decision Tree - Banknote Authentication
##################################################    
def run_dt_bank():
    X,Y,trainX,testX, trainY, testY = get_bank_train_test()            
    depth = range(1,11) + [15,20 ,30, 50,100]
    params = { 'criterion' : ['entropy','gini'],
               'max_depth' : depth ,
               'min_samples_split' : [2,4,6,10],
               'splitter' : ["best", "random"] }
                  
    clf = tree.DecisionTreeClassifier(random_state=100)                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "DT_bank" ,False)  
    print('run_dt : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_dt : Test accuracy: %.3f' % best_clf.score(testX, testY))
                                                  
    #plot validation and learning curve score
    plot_dt_gs(res_df, 'Bank -')
    plot_learning_curve(best_clf, "DT_bank : Learning Curve", X,Y ).show()
      
    #plot validation and learning curve time
    plot_validation_time(res_df ,"DT_bank : fit/score time").show()
    plot_learning_time(X,Y,best_clf ,"DT_bank : Time Complexity", ylim=(-0.001, 0.003)).show()

    get_f1score(best_clf,trainX,trainY,testX,testY, "SVM_bankAuth",True, True)
    
##################################################
# Decision Tree - Phishing
##################################################    
def run_dt_phishing():
    X,Y,trainX,testX, trainY, testY = get_phishing_train_test()           
    depth = range(1,11) + [15,20 ,30, 50,100]
    params = { 'criterion' : ['entropy','gini'],
               'max_depth' : depth ,
               #'min_samples_split' : [2,4,6,10],
               'splitter' : ["best", "random"] }
    if(BEST):
        params = { 'criterion' : ['entropy'],'max_depth' : [10] ,'splitter' : ["best"] }
              
    clf = tree.DecisionTreeClassifier(random_state=100)                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "DT_phishing" ,False)  
    print('run_dt : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_dt : Test accuracy: %.3f' % best_clf.score(testX, testY))
                                                  
   # plot_dt_gs(res_df, 'Phishing -')
   
    plot_learning_curve(best_clf, "DT_phishing: Learning Curve", X,Y).show()
    plot_learning_time(X,Y,best_clf ,"DT_phishing : Time Complexity", ylim=(-0.001, 0.006)).show()
    get_f1score(best_clf,trainX,trainY,testX,testY, "SVM_phishing",True, True)
    


##################################################
# Decision Tree - MAIN
##################################################
if __name__ == "__main__":   
    print "Decision Tree main .."
    
#    run_dt_bank()
#    run_dt_abalone()
#    run_dt_phishing()

































#Abalone 
#{'min_samples_split': 2, 'splitter': 'best', 'criterion': 'gini', 'max_depth': 4}
#Train accuracy: 0.711
#Test accuracy: 0.715
#run_dt : Train accuracy: 0.711
#run_dt : Test accuracy: 0.715