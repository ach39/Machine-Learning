# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 14:16:18 2017

@author: ac104q
"""

from ac_util import *

ann_cv = ms.ShuffleSplit(n_splits=40, test_size=0.2, random_state=0)
##################################################
# Neural Net - Iris
##################################################    
def run_ann_iris():
    X,Y,trainX,testX, trainY, testY = get_iris_train_test()
    
    #d = X.shape[1]
    #hidden_l = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
    hidden_l = [(10,), (20,), (10,10), (20,20), (10,20), (20,10) ]
    alpha = [.0001,.001,.01,.1,1,1.5]
    
    params = {'activation': ['relu','logistic'],
              'alpha': alpha,
              'hidden_layer_sizes':hidden_l,
              'solver' : [ 'sgd']   # ‘lbfgs’, ‘sgd’, ‘adam’
              }
    
    #clf = MLPClassifier(random_state=100,early_stopping=True,max_iter=1000)  
    clf = MLPClassifier(random_state=100,learning_rate_init=0.01, max_iter=500)                                                    
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "ANN_iris", False)                                                    
    
    print('run_ann : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_ann : Test accuracy: %.3f' % best_clf.score(testX, testY))
        
    #plot validation and learning curve score
    cv = ms.ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    plot_learning_curve(best_clf, "ANN_Iris : LC", X,Y, cv=cv).show()
    
    #plot validation and learning curve time
    
    ylim=(-0.05 , 0.07)    
    #plot_validation_time(res_df ,"BOOST_Iris : fit/score time",ylim).show()
    plot_learning_time(X,Y,best_clf ,"ANN_Iris : LearningCurve time",ylim).show()
  
  
##################################################
# Neural Net - abalone
##################################################
def run_ann_abalone():
    X,Y,trainX,testX, trainY, testY = get_abalone_train_test()
    d = X.shape[1]
    hidden_l = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
    #hidden_l = [(10,), (20,), (10,10), (20,20), (10,20), (20,10) ]
    alpha = [ 0.0001, 0.001, 0.01, 0.1, 1, 2]

    params = {'activation': ['relu' ,'logistic'],
              'alpha': alpha,
              'hidden_layer_sizes':hidden_l,
              'solver' : ['lbfgs' ,'adam']   # ‘lbfgs’, ‘sgd’, ‘adam’
              #'max_iter' : [200,400,800]
              }

#    params = {'activation': ['relu'],
#          'alpha': [1],
#          'hidden_layer_sizes':[(3,), (5,), (7,), (10,)],
#          'solver' : ['lbfgs'],
#          'max_iter' : [10,50, 100,200,400,800,1000,1500]  }
#   clf = MLPClassifier(random_state=100)    
   

                            
    if(BEST) :
        params = {'activation': ['logistic'],
          'alpha': [1],
          'hidden_layer_sizes':[(5,)],
          'solver' : ['lbfgs']}
        
    clf = MLPClassifier(random_state=100,early_stopping=True,max_iter=2000)  
                                                       
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "ANN_abalone", False)                                                    
    
    print('run_ann : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_ann : Test accuracy: %.3f' % best_clf.score(testX, testY))
        
    plot_learning_curve(best_clf, "ANN_abalone : LearningCurve", X,Y,cv=ann_cv).show()
    plot_learning_time(X,Y,best_clf ,"ANN_abalone : TimeComplexity",ylim=(-0.05,2)).show()
    get_f1score(best_clf,trainX,trainY,testX,testY, "ANN_abalone",False)
    
##################################################
# Neural Net - bankNote authentication
##################################################    
def run_ann_bank():
    X,Y,trainX,testX, trainY, testY = get_bank_train_test()
    
    d = X.shape[1]
    hidden_l = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
    #hidden_l = [(10,), (20,), (10,10), (20,20), (10,20), (20,10) ]
    #alpha = [.0001,.001,.01,.1,1,1.5]  #L2 penalty
    alpha = [ 0.0001, 0.001, 0.01, 0.1, 1, 5]

    params = {'activation': ['relu' ,'logistic'],
              'alpha': alpha,
              'hidden_layer_sizes':hidden_l,
              'solver' : [ 'adam' ,'lbfgs'],   # ‘lbfgs’, ‘sgd’, ‘adam’
              }
    
    clf = MLPClassifier(random_state=100,early_stopping=True , max_iter=1500)                                                    
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "ANN_bank", False)                                                    
    
    print('run_ann : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_ann : Test accuracy: %.3f' % best_clf.score(testX, testY))
        
    #plot validation and learning curve score
    cv = ms.ShuffleSplit(n_splits=40, test_size=0.2, random_state=0)
    plot_learning_curve(best_clf, "ANN_bank : LearningCurve", X,Y, cv=cv).show()
    
    #plot validation and learning curve time
    
    ylim=(-0.05 , 0.4)    
    plot_learning_time(X,Y,best_clf ,"ANN_bank : TimeComplexity",ylim).show()    
    get_f1score(best_clf,trainX,trainY,testX,testY, "ANN_bank",True, True)
    
##################################################
# Neural Net - Phishing
##################################################    
def run_ann_phishing():
    X,Y,trainX,testX, trainY, testY = get_phishing_train_test()
    
    d = X.shape[1]
    hidden_l = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
    #hidden_l = [(10,), (20,), (10,10), (20,20), (10,20), (20,10) ]
    #alpha = [.0001,.001,.01,.1,1,1.5]  #L2 penalty
    alpha = [ 0.0001, 0.001, 0.01, 0.1, 1, 5]

    params = {'activation': ['relu' ,'logistic'],
              'alpha': alpha,
              'hidden_layer_sizes':hidden_l,
              'solver' : [ 'adam' ,'lbfgs'],   # ‘lbfgs’, ‘sgd’, ‘adam’
              }

    params = {'activation': ['relu'],
          'alpha': [0.1],
          'hidden_layer_sizes':[(10,), (15,), (20,), (30,)],
          #'hidden_layer_sizes':[(40,), (50,)],
          'solver' : ['lbfgs'],
          'max_iter' : [10,50, 100,200,400,800,1000,1500]  }
    clf = MLPClassifier(random_state=100)   
   
    if(BEST) :
         params = {'activation': ['relu'],'alpha': [0.1],'hidden_layer_sizes':(30L,),
              'solver' : ['lbfgs'] }  #<- best
          
    #clf = MLPClassifier(random_state=100,early_stopping=True , max_iter=1000)                                                    
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "ANN_phishing", False)                                                    
    
    print('run_ann : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_ann : Test accuracy: %.3f' % best_clf.score(testX, testY))
        
    plot_learning_curve(best_clf, "ANN_phishing : LearningCurve", X,Y,cv=ann_cv).show()      
    plot_learning_time(X,Y,best_clf ,"ANN_phishing : TimeComplexity",ylim=(-0.05 , 3)).show()            
    get_f1score(best_clf,trainX,trainY,testX,testY, "ANN_phishing",True, True)


##################################################
# Neural Net - main
##################################################
if __name__ == "__main__" :
    print "Nueral Net ... "

#    run_ann_bank()      
#    run_ann_abalone()
#   run_ann_phishing()








#Phishing
#{'alpha': 0.1, 'activation': 'relu', 'solver': 'lbfgs', 'hidden_layer_sizes': (30L,)}

# Bank 2 ,4 8 neurons
#{'alpha': 0.0001, 'activation': 'relu', 'solver': 'lbfgs', 'hidden_layer_sizes': (4L,)}
#Train accuracy: 1.0000
#Test accuracy: 1.0000



#{'alpha': 1, 'activation': 'logistic', 'solver': 'lbfgs', 'hidden_layer_sizes': (5L, 5L)}
#Train accuracy: 0.7318
#Test accuracy: 0.7329
#[ 0.75149701  0.74700599  0.75149701  0.74401198  0.74491018  0.74401198
#  0.74165954  0.74026946  0.7411843   0.73929961]
#
#[ 0.72009569  0.74521531  0.75239234  0.76196172  0.76435407  0.76196172
#  0.76196172  0.76196172  0.76435407  0.76196172]
  

# Abalone

#{'alpha': 0.001, 'activation': 'relu', 'max_iter': 200, 'solver': 'adam', 'hidden_layer_sizes': (20L, 20L)}
#Train accuracy: 0.7342
#Test accuracy: 0.7289


#{'alpha': 1, 'activation': 'relu', 'solver': 'sgd', 'hidden_layer_sizes': (20L, 20L, 20L)}
#Train accuracy: 0.734
#Test accuracy: 0.741

#{'alpha': 0.01, 'activation': 'relu', 'solver': 'adam', 'hidden_layer_sizes': (14L,)}
#Train accuracy: 0.726
#Test accuracy: 0.726



#{'alpha': 0.001, 'activation': 'relu', 'max_iter': 200, 'solver': 'adam', 'hidden_layer_sizes': (20L, 20L)}
#Train accuracy: 0.734
#Test accuracy: 0.729

#best score - Iris
    #{'alpha': 0.0001, 'activation': 'relu', 'solver': 'sgd', 'hidden_layer_sizes': (20, 10)}
    #Train accuracy: 0.975
    #Test accuracy: 1.000