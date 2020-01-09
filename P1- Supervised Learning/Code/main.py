# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:42:02 2018

@author: ac104q
"""
from sys import argv
from ac_KNN import *
from ac_DTree import *
from ac_Boost import *
from ac_SVM import *
from ac_ANN import *
import warnings
warnings.filterwarnings("ignore")

 
def help_p1() :
    instructions =  "\n\n\
    ------------------------------------------------------\n \
    Enter the number to run desired supervised algorithm. \n \
        Eg: Enter 1 to run KNN for ABALONE dataset \n \
            Enter 4 to SVM for PHISHING dataset \n \
            Enter 11 to run all five classifiers for ABALONE dataset \n\n \
    \
    1.  run_knn_abalone() \n \
    2.  run_knn_phishing() \n \
    3.  run_svm_abalone()  \n \
    4.  run_svm_phishing() \n \
    5.  run_adaboost_abalone() \n \
    6.  run_adaboost_phishing() \n \
    7.  run_dt_abalone() \n \
    8.  run_dt_phishing() \n \
    9.  run_ann_abalone() \n \
    10. run_ann_phishing() \n \
    11. run_all_abalone() \n \
    12. run_all_phishing() \n \
    ------------------------------------------------------ "
    #for i,v in enumerate(myfuncs):print i, ". ", str(myfuncs[i])
    print instructions


def run_all_phishing():
    run_knn_phishing()
    run_svm_phishing()
    run_dt_phishing()
    run_adaboost_phishing()
    run_ann_phishing()
    
  
def run_all_abalone():
    run_knn_abalone()
    run_svm_abalone()
    run_dt_abalone()
    run_adaboost_abalone()
    run_ann_abalone()

option=0

myfuncs=[help_p1,
         run_knn_abalone,   run_knn_phishing,
         run_svm_abalone,   run_svm_phishing,
         run_adaboost_abalone, run_adaboost_phishing,
         run_dt_abalone,    run_dt_phishing,
         run_ann_abalone,   run_ann_phishing,
         run_all_abalone,   run_all_phishing ]

         
if __name__ == "__main__":
    
    if len(argv) == 2:
        option = int(argv[1])
        if (option >=0) and (option <=12) :
            print "\n Running option#", option , "\n"
            myfuncs[option]()
        else :
            print "\n Error  : please specify argument in range 1-12.\n"
  
    else:
        help_p1()
    


 