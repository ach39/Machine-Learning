CS 7641 Machine Learning: Project-3
[Arti Chauhan:  Mar-23-2018]

------------------------------------------------------------------------------------

1. This assignment was implemented in python using Scikit Learn.

2. Relevant files to reproduce the results can be found in folder 'code'

3. Abalone and Phishing datasets from Project-1 were reused for this assignment.

4. Due to lack of space in report, charts and cluster-visualiztion graphs are shrunk 
   in size.Better resolution graphs are provided in CHARTS_P3.xlsx

----------------------------------------------------------------------------------
How to run code ?


Part-1 - Clustering analysis on original data.

	 cmd :  python ac_clustering.py ORG


Part-2 - Dimensionality reduction - produces dimesion-reduced dataset in respective folders (PCA,ICA,RF,RP)

 	 cmd : 	python ac_pca.py
			python ac_ica.py
			python ac_rand_proj.py
			python ac_rand_forest.py


Part-3 - Clustering-analysis on data produced in part-2.

	 cmd :  python ac_clustering.py PCA
			python ac_clustering.py ICA
			python ac_clustering.py RP
			python ac_clustering.py RF


Part-4 - Neural network analysis on produced in part-2.
 	 Uncomment following functions in corresponding files and run commands listed in Part-2 above.
	 These NN evaluation functions are commented on purpose as it takes long time finish.

 	 cmd : 	ac_pca.py			-> run_pca_nn()
			ac_ica.py 		-> run_ica_nn()
			ac_rand_proj.py 	-> run_rp_nn()
			ac_rand_forest.py 	-> run_rf_nn() 



Part-5 - Neural network analysis using Clustering as dimensionality reduction technique.

	Uncomment run_NN_phishing() in file ac_clustering.py and run 
	cmd :  python ac_clustering.py ORG




----------------------------------------------------------------------------------

