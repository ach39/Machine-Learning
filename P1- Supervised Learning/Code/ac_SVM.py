# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 13:45:06 2017

@author: ac104q
"""
from ac_util import *

##################################################
# SVM Validation curve
#gamma = [0.01, 0.1,1.0,5.0, 10.0]
##################################################
#from ggplot import *
   
def plot_svc_gs(res_df):

    df = res_df[res_df.param_kernel=='linear']   
    plot_it(df ,'param_C', "kernel = LINEAR ").show()

    df = res_df[(res_df.param_kernel=='rbf') & (res_df.param_gamma==0.5)]
    plot_it(df ,'param_C', "kernel/gamma = RBF /0.5").show()

    df = res_df[(res_df.param_kernel=='rbf') & (res_df.param_gamma==2)]
    plot_it(df ,'param_C', "kernel/gamma = RBF /2").show()
    
    df = res_df[(res_df.param_kernel=='rbf') & (res_df.param_gamma==1.0)]
    plot_it(df ,'param_C', "kernel/gamma = RBF /1").show()

    df = res_df[(res_df.param_kernel=='rbf') & (res_df.param_gamma==1.5)]
    plot_it(df ,'param_C', "kernel/gamma = RBF /1.5").show()
#
#    df = res_df[(res_df.param_kernel=='rbf')]
#    plot_it(df ,'param_C', "kernel/gamma = RBF /auto").show()
    
##################################################
# SVM - iris
##################################################
def run_svm_iris():
    X,Y,trainX,testX, trainY, testY = get_iris_train_test()
    
    param_range = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 1.5 ,5, 10.0, 100.0]
    param_range_lin = [ 0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5 ,3]
    
    params = [{'C': param_range_lin, 'kernel': ['linear']},
              {'C': param_range, 'gamma': param_range,'kernel': ['rbf']}
             ]
                                                      
    clf = SVC()                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "SVM_iris")  
    print('run_svm : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_svm : Test accuracy: %.3f' % best_clf.score(testX, testY))
                                                  
    #plot validation and learning curve score
    plot_svc_gs(res_df)
    plot_learning_curve(best_clf, "SVM_Iris : LC", X,Y ).show()
      
    #plot validation and learning curve time
    plot_validation_time(res_df ,"SVM_Iris : fit/score time").show()
    plot_learning_time(X,Y,best_clf ,"SVM_Iris : LearningCurve time").show()


##################################################
# SVM - Abalone
##################################################
def run_svm_abalone():

    X,Y,trainX,testX, trainY, testY = get_abalone_train_test()
    
    C_rbf = [0.01, 0.1, 1.0, 2.0,5.0,10]
    gamma = [ 0.5, 1.0, 1.5, 3.0, 5.0]
    
    C_lin = [ 0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5 ,3,4.5,6,10]
    #C_lin = [ 0.01, 0.1,1,10]
    
    params = [ {'C': C_lin, 'kernel': ['linear']},
              #{'C': C_rbf, 'kernel': ['poly'] ,'gamma': gamma , 'degree' :[3,5,7,9]},
              #{'C': C_rbf, 'kernel': ['rbf']},         #<- to test gamma=auto
              {'C': C_rbf, 'gamma': gamma,'kernel': ['rbf']}  ]
    
    if(BEST) : params = [ {'C': [2], 'kernel': ['rbf'] , 'gamma' :[1.5] } ]
      
    clf = SVC()                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "SVM_abalone")  
    print('run_svm : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_svm : Test accuracy: %.3f' % best_clf.score(testX, testY))
                                                  
    #plot_svc_gs(res_df)

    plot_learning_curve(best_clf, "SVM_abalone : Learning Curve", X,Y ).show()      
    plot_learning_time(X,Y,best_clf ,"SVM_abalone : Time Complexity",ylim=(-0.001 , 0.5)).show()
    get_f1score(best_clf,trainX,trainY,testX,testY, "SVM_bankAuth",False, True)

##################################################
# SVM - Banknote Authentication
##################################################
def run_svm_bank(): 
    X,Y,trainX,testX, trainY, testY = get_bank_train_test()

    C_rbf = [0.01, 0.1, 1.0, 2.0, 10.0]
    gamma = [0.01, 0.1,1.0 ,5.0, 10.0]    
    C_lin = [ 0.001, 0.01, 0.1, 0.5, 1.0, 3 ,5, 7, 10 , 30]
    
    params = [
               {'C': C_lin, 'kernel': ['linear']},
              #{'C': C_rbf, 'kernel': ['rbf']}
              {'C': C_rbf, 'gamma': gamma,'kernel': ['rbf']}    
             ]
                                                      
    clf = SVC()                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "SVM_bank")  
    print('run_svm : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_svm : Test accuracy: %.3f' % best_clf.score(testX, testY))
                                                  
    #plot validation and learning curve score
    #plot_svc_gs(res_df)
    cv = ms.ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(best_clf, "SVM_bank : Learning Curve", X,Y, cv=cv ).show()
      
    plot_learning_time(X,Y,best_clf ,"SVM_bank : Time Complexity",ylim=(-0.01 , 0.05)).show()

    get_f1score(best_clf,trainX,trainY,testX,testY, "SVM_bankAuth",True, True)


##################################################
# SVM - Phisihing
##################################################
def run_svm_phishing(): 
    X,Y,trainX,testX, trainY, testY = get_phishing_train_test()

    C_rbf = [0.01, 0.1, 1.0, 2.0, 10.0,15, 20]
    gamma = [0.01, 0.1,1.0 ,5.0, 10.0]    
    C_lin = [ 0.001, 0.01, 0.1, 0.5, 1.0, 3 ,5, 7, 10 , 30]
    
    params = [
              {'C': C_lin, 'kernel': ['linear']},
              #{'C': C_rbf, 'kernel': ['poly'] ,'gamma': gamma , 'degree' :[3,5,7,9]},
              {'C': C_rbf, 'gamma': gamma ,'kernel': ['rbf']}    # gamma auto gives best results
             ]
      
    if(BEST) : params = [{'C': [10], 'gamma': [0.1] ,'kernel': ['rbf']} ]   
                                           
    clf = SVC()                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "SVM_phishing")  
    print('run_svm : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_svm : Test accuracy: %.3f' % best_clf.score(testX, testY))
                                                  
    # plot_svc_gs(res_df)
  
    cv = ms.ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
    plot_learning_curve(best_clf, "SVM_phishing : Learning Curve", X,Y, cv=cv ).show()
    plot_learning_time(X,Y,best_clf ,"SVM_phishing : Time Complexity",ylim=(-0.001, 0.07)).show()
    get_f1score(best_clf,trainX,trainY,testX,testY, "SVM_phishing",True, True)



##################################################
# SVM - MAIN 
##################################################
if __name__=="__main__" :
    print "SVM main ..."
    
#    run_svm_bank()
#    run_svm_abalone()
#    run_svm_phishing()



 
 


# Abalone
#{'kernel': 'rbf', 'C': 1.0, 'gamma': 1.0}
#Train accuracy: 0.733
#Test accuracy: 0.730


#{'kernel': 'rbf', 'C': 2.0, 'gamma': 1.0}
#Train accuracy: 0.743
#Test accuracy: 0.731
