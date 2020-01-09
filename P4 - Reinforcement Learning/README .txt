CS 7641 Machine Learning: Project-4
[Arti Chauhan:  Apr-13-2018]

------------------------------------------------------------------------------------

1. This assignment was implemented using Burlap library and Jython.

2. Relevant files to reproduce the results can be found under 'code' directory.

3. logs for each MDPs are generated under following locations.
 	a) Labyrinth MDP :	
		code/LYB_logs/VI/*.csv
		code/LYB_logs/PI/*.csv
		code/LYB_logs/QI/*.csv

 	b) Treasure Hunt MDP :	
		code/TH_logs/VI/*.csv
		code/TH_logs/PI/*.csv
		code/TH_logs/QI/*.csv

----------------------------------------------------------------------------------
How to run code ?

1. Install Jython
2. Burlap.jar is included in the submission.
3. Go to 'code' folder and execute run.bat from command prompt.Please note batch file assumes path to jython is
	C:\jython2.7.0\bin\jython. If that's not true for your setup then modify the path.

	C:\jython2.7.0\bin\jython ac_lybrinth.py 	<- runs VI,PI and QL for labyrinth MDP 
   	C:\jython2.7.0\bin\jython ac_treasureHunt.py 	<- runs VI,PI and QL for Treasure Hunt MDP 


----------------------------------------------------------------------------------

