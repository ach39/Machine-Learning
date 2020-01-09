package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 *  AC  2/6/2018.
 *  Dataset size : 2456 , 70% = 1720
 */
 
public class ac_PhishingTest_RHC {
	
    private static Instance[] instances = initializeInstances();
	private static Instance[] train_set = Arrays.copyOfRange(instances,0,1720);
	private static Instance[] test_set  = Arrays.copyOfRange(instances,1720, 2456);
	
    private static DataSet set = new DataSet(train_set);
	
    private static int inputLayer = 30, hiddenLayer = 30, outputLayer = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
	private static ErrorMeasure measure = new SumOfSquaresError();
    
	private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"RHC"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
		
		String final_result = "";
        String my_str = "";
		
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }


        int [] ITER = {10, 100, 500, 1000, 2500 ,5000, 8000};

		System.out.println("RHC, iter, train_acc, test_acc, train-time, test-time ");
		
		for (int train_it : ITER) {
			oa[0] = new RandomizedHillClimbing(nnop[0]);
			double start = 0, end=0, trainingTime=0, testingTime=0, correct=0, incorrect=0,train_acc=0, test_acc=0;
		
			/***********  train the model : RHC  ******************/
            start = System.nanoTime();

			for(int j=0 ; j <train_it ;j++){
				oa[0].train(); }

            //train(oa[0], networks[0], oaNames[0],train_it); //trainer.train();

			end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[0].getOptimal();
            networks[0].setWeights(optimalInstance.getData());

			/************ Get Train Accuracy *****************/
            double predicted, actual;
            start = System.nanoTime();
            
			for(int j = 0; j < train_set.length; j++) {
                networks[0].setInputValues(train_set[j].getData());
                networks[0].run();
                actual = Double.parseDouble(train_set[j].getLabel().toString());
                predicted = Double.parseDouble(networks[0].getOutputValues().toString());
                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);
			train_acc = 100 * correct/(correct + incorrect);

			/************ Get Test Accuracy *****************/
			correct= 0;
			incorrect = 0;
			start = System.nanoTime();
			
			for (int j = 0; j < test_set.length; j++) {
                networks[0].setInputValues(test_set[j].getData());
                networks[0].run();
                actual = Double.parseDouble(test_set[j].getLabel().toString());
                predicted = Double.parseDouble(networks[0].getOutputValues().toString());
                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
			end = System.nanoTime();
			testingTime = end-start;
			testingTime /= Math.pow(10,9);
			test_acc = 100 * correct/(correct + incorrect);
			
            /***********  Print Results  ******************/
            results = oaNames[0] + ", " + train_it + ", " + df.format(train_acc) + ", " + df.format(test_acc) + ", " +
                        df.format(trainingTime) + ", " + df.format(testingTime) + "\n";
            System.out.print(results);
		}
        
        //System.out.println(results);
    }


 private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");
        int trainingIterations = iteration;
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double train_error = 0;
            for(int j = 0; j < train_set.length; j++) {
                network.setInputValues(train_set[j].getData());
                network.run();

                Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += measure.value(output, example);
            }


            //System.out.println(df.format(train_error));
        }
    }


    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[2456][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/phishing_ac_p2.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[30]; // 30 attributes
                attributes[i][1] = new double[1];  // labels

                for(int j = 0; j < 30; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0] < 0 ? 0 : 1));
        }

        return instances;
    }
}
