# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 14:41:49 2017
@author: ac104q

"""

from ac_util import *


##################################################
# KNN Validation curve
##################################################   
def plot_knn_gs(res_df) :
    '''
    mean_fit_time 	mean_score_time	
    mean_test_score	mean_train_score	
    param_metric	     param_n_neighbors
    param_weights    
    std_fit_time	      std_score_time	
    std_test_score	std_train_score

    '''
    df = res_df[(res_df.param_metric=='euclidean') & (res_df.param_weights=='uniform')]   
    plot_it(df ,'param_n_neighbors', "weight/metric = Uniform / Euclidean").show()

    df = res_df[(res_df.param_metric=='manhattan') & (res_df.param_weights=='uniform')]
    plot_it(df ,'param_n_neighbors', "weight/metric = Uniform / Manhtttan").show()

     
    df = res_df[(res_df.param_metric=='euclidean') & (res_df.param_weights=='distance')]   
    plot_it(df ,'param_n_neighbors', "weight/metric = Distance / Euclidean ").show()

    df = res_df[(res_df.param_metric=='manhattan') & (res_df.param_weights=='distance')]
    plot_it(df ,'param_n_neighbors', "weight/metric = Distance / Manhtttan ").show()
                                       
    df = res_df[(res_df.param_metric=='minkowski') & (res_df.param_weights=='uniform')]   
    plot_it(df ,'param_n_neighbors', "weight/metric = Uniform / minkowski").show()

    df = res_df[(res_df.param_metric=='minkowski') & (res_df.param_weights=='uniform')]
    plot_it(df ,'param_n_neighbors', "weight/metric = Uniform / minkowski").show()

                                                                            
##################################################
# KNN - Iris
##################################################
def run_knn_iris():
    X,Y,trainX,testX, trainY, testY = get_iris_train_test()
        
    #params = {'metric':['manhattan','euclidean','chebyshev'],
    params = {'metric':['manhattan','euclidean'],   
              'n_neighbors':np.arange(1,30,5),
              'weights':['uniform'] #,'distance'
              }  
                                                  
    clf = KNeighborsClassifier()                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "knn_iris")                                                    
    
    print('run_knn : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_knn : Test accuracy: %.3f' % best_clf.score(testX, testY))
    
    #plot validation and learning curve score
    plot_knn_gs(res_df)
    
    #ts = np.linspace(.3, .8, 7) 
    plot_learning_curve(best_clf, "KNN_Iris : LC ", X,Y, train_sizes=ts).show()
    
    #plot validation and learning curve time
    plot_validation_time(res_df ,"KNN_Iris : fit/score time").show()
    plot_learning_time(X,Y,best_clf ,"KNN_Iris : LearningCurve time").show()
    
 
##################################################
# KNN - Abalone 
##################################################
def run_knn_abalone():
    X,Y,trainX,testX, trainY, testY = get_abalone_train_test()
    params = {'metric': ['manhattan','euclidean','minkowski'],
              'n_neighbors':[1,5,10,20,30,40,50, 80,200],
              'weights':['uniform', 'distance' ] 
              }                                                    
    if(BEST):
       params = {'metric':['euclidean'],   'n_neighbors':[40],'weights':['uniform'] } 
    
    clf = KNeighborsClassifier()                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "knn_abalone")                                                    
    
    print('run_knn : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_knn : Test accuracy: %.3f' % best_clf.score(testX, testY))
                
    #plot_knn_gs(res_df)
    
    plot_learning_curve(best_clf, "KNN_Abalone : Learning Curve ", X,Y).show()
    plot_learning_time(X,Y,best_clf ,"KNN_Abalone : Time Complexity",ylim=(-0.001 , 0.055)).show()
    get_f1score(best_clf,trainX,trainY,testX,testY, "KNN-Abalone",False)
    
    
##################################################
# KNN - Banknote Authentication
##################################################
def run_knn_bank(): 
    X,Y,trainX,testX, trainY, testY = get_bank_train_test()
    params = {'metric':['manhattan','euclidean'],   
              'n_neighbors':[3,5,10,20,30,50,100,200],
              'weights':['uniform','distance']
              }                                                  
    clf = KNeighborsClassifier()                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "knn_bankAuth")                                                    
    
    print('run_knn : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_knn : Test accuracy: %.3f' % best_clf.score(testX, testY))
    
   # plot validation and learning curve score
   #plot_knn_gs(res_df)
    
    plot_learning_curve(best_clf, "KNN_bank : Learning Curve ", X,Y).show()
    plot_learning_time(X,Y,best_clf ,"KNN_bank : Time complexity",ylim=(-0.001 , 0.01)).show()
    get_f1score(best_clf,trainX,trainY,testX,testY, "KNN_bank",False)
    
##################################################
# KNN - phishing
##################################################
def run_knn_phishing(): 
    X,Y,trainX,testX, trainY, testY = get_phishing_train_test()
    params = {'metric':['manhattan','euclidean' ,'minkowski'],   
              'n_neighbors':[1,2,3,4,5,6,8,10,20,30,50,100,200],
              'weights':['uniform'] #,'distance']
              } 

    if(BEST):
        params = {'metric':['manhattan'],   'n_neighbors':[6],'weights':['distance'] } 
                                                             
    clf = KNeighborsClassifier()                                                   
    best_clf , res_df = tuneParams(clf, trainX, trainY, testX, testY, params, "knn_phishing")                                                    
    
    print('run_knn : Train accuracy: %.3f' % best_clf.score(trainX, trainY))
    print('run_knn : Test accuracy: %.3f' % best_clf.score(testX, testY))
    
   #plot validation and learning curve score
    #plot_knn_gs(res_df)
    
    plot_learning_curve(best_clf, "KNN_phishing : Learning Curve ", X,Y).show()
    plot_learning_time(X,Y,best_clf ,"KNN_phishing : Time complexity",ylim=(-0.002 , 0.08)).show()
    get_f1score(best_clf,trainX,trainY,testX,testY, "KNN_phishing",True)
    
            
##################################################
# KNN - MAIN
##################################################

if __name__ == "__main__":   
    print "KNN main .."
    
    #run_knn_bank()    
#    run_knn_abalone()
#    run_knn_phishing()
  
  
  
  




















#phishing
#{'n_neighbors': 10, 'metric': 'manhattan', 'weights': 'uniform'}


#def my_test(best_clf):
#    digit=5
#    f = "datasets/banknote_auth_shuffled_test.csv"
#    df = pd.read_csv(f, index_col=0) 
#    df_array = df.as_matrix()
#    test_x = df_array[:,:4]
#    test_y = df_array[:,-1]
#    
#    y_pred   = best_clf.predict(test_x) 
#    test_acc = round(accuracy_score(test_y, y_pred ), digit)
#    f1       = round(f1_score(test_y, y_pred, average='weighted'), digit)
#    recall   = round(recall_score(test_y, y_pred, average='weighted'), digit)
#    prec     = round(precision_score(test_y, y_pred, average='weighted'), digit)  
#    
#    print ('Test_Acc  \t F1 score \t Recall \t Precision')                                  
#    print  test_acc ,'\t', f1,'\t', recall ,'\t', prec
#
#    df['pred']= y_pred


 #Abalone  - overfitting                                                 
#{'n_neighbors': 30, 'metric': 'euclidean', 'weights': 'distance'}
#Train accuracy: 1.000
#Test accuracy: 0.728

#{'n_neighbors': 40, 'metric': 'euclidean'}  weights=uniform
#Train accuracy: 0.725
#Test accuracy: 0.730


    
#    plt.fill_between(x, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#    plt.fill_between(x, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # param_metric = 'euclidean' , ,manhattan,
    # param_weights = 'uniform' , 'distance'