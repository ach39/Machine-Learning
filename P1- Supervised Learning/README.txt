CS 7641 Machine Learning: Project-1
[Arti Chauhan:  Feb-3-2018]

------------------------------------------

Directory structure 

1. code/output/*  : has data for hyperparameters tuning experiments, LearningCurve and TimeComplexity

	1a. Results of all experiments to tune hyperparameters for different classifier can be found in 
		code/output/Tune/ClassifierName_datasetName.xlx 

	1b. Data to produce LearningCurve for different classifier can be found in 
		code/output/LearningCurve/ClassifierName_datasetName_LC_score.xlx 

	1c. Data for time taken to train/test by best_model can be found in 
		code/output/TimeComplexity/ClassifierName_datasetName_timing.xlx 
		
		where classifierName take on values [KNN,SVM,DT,BOOST,ANN]
		datasetName take on values [Abalone, Phishing]

		
2. code/dataset/*  : has csv for Abalone and Phishing dataset.


3. code/*.py : has pyhton code files : 
	3a. ac_ClassifierName.py - contains code to tune,train & test different classifier. There's one file for each classifier. 
	3b. ac_util.py - contains helper functions.
	3c. main.py - driver code to run experiments
	
	
------------------------------------------

How to run the code ?	


This project was implement in python v2.7 (Sklearn- 0.19.0).
Results for any classifier can be reproduced by running following from main.py 

command : main.py <argument>
	where <argument> is a number between 1-12.

Eg: Enter 1 to run KNN for abalone dataset 
    Enter 4 to SVM for Phishing dataset 
    Enter 11 to run all classifiers for Phishing dataset 

1.  run_knn_abalone() 
2.  run_knn_Phishing() 
3.  run_svm_abalone()  
4.  run_svm_Phishing() 
5.  run_adaboost_abalone() 
6.  run_adaboost_Phishing() 
7.  run_dt_abalone() 
8.  run_dt_Phishing() 
9.  run_ann_abalone() 
10. run_ann_Phishing() 
11. run_all_abalone()
12. run_all_Phishing() 

Important Note - By default,above models run using single (ie best) set of hyperparameters.
If you would like to reproduce all grid-search results,set 'BEST' variable in ac_util.py(line 53) to zero.
Please note that this will take longer time to complete.

------------------------------------------ 	