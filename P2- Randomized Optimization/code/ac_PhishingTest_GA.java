package opt.test;

/**
 * AC  2/6/2018.
 *  Dataset size : 2456 , 70% = 1720
 */

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import func.nn.activation.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

public class ac_PhishingTest_GA {
    private static Instance[] instances = initializeInstances();
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, 1720);
    private static Instance[] test_set = Arrays.copyOfRange(instances, 1720, 2456);

    private static DataSet set = new DataSet(train_set);

    private static int inputLayer = 30, hiddenLayer=30, outputLayer = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"GA"};
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
		
        int[] ITER = {10, 100, 500, 1000, 2500, 5000 ,8000};
        //int[] ITER = {50};
   	
		int[] pop_sz 	= {10, 50, 100, 200, 500};
        //int[] pop_sz 	= {200};
        //int[] mate 		= {5 , 10, 25, 50,  120, 300};
        //int[] mutate 	= {2 , 5,  8 , 10,  20,  25};

        //int[] mutate 	= {.01, 0.02, 0.04 , 0.05,  0.07 , 0.09};

        System.out.println("GA, iter, pop_sz, mate,mutate, train_acc, test_acc, train-time, test-time Run");

        int mate=0, mutate=0;

        for (int train_it : ITER) {
		
            for(int i=0 ; i<pop_sz.length ; i++) {
                mate   = (int)(0.6 * pop_sz[i]);   // mating prob = 60%
                mutate = (int)(0.01 * pop_sz[i]);  // mutating prob = 3%
                oa[0] = new StandardGeneticAlgorithm(pop_sz[i],mate, mutate, nnop[0]);

                double start=0, end= 0, trainingTime=0, testingTime=0, correct= 0, incorrect= 0, train_acc= 0, test_acc = 0;

                /***********  train the model : GA ******************/
                start = System.nanoTime();
                for (int j = 0; j < train_it; j++) {
                    oa[0].train();
                }
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[0].getOptimal();
                networks[0].setWeights(optimalInstance.getData());

                /************ Train set Accuracy *****************/
                double predicted=0, actual=0;
                start = System.nanoTime();
                for (int j = 0; j < train_set.length; j++) {
                    networks[0].setInputValues(train_set[j].getData());
                    networks[0].run();

                    actual = Double.parseDouble(train_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[0].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);
                train_acc = 100 * correct/(correct + incorrect);


                /***********  Test Set Accuracy ******************/
                correct = 0;
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
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);
                test_acc = 100 * correct/(correct + incorrect);

                /***********  Print Results  ******************/
                results = oaNames[0] + ", " + train_it + ", " + pop_sz[i] + ", " + mate + ", " + mutate + ", "+ df.format(train_acc) + ", " + df.format(test_acc) + ", " +
                        df.format(trainingTime) + ", " + df.format(testingTime) + " , 2 \n";
                System.out.print(results);

            } //(pop,mate,mutate)

        } //Iterations

    }//main

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[2456][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/phishing_ac_p2.csv")));

            //for each sample
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[30]; // 30 attributes
                attributes[i][1] = new double[1];  // labels +1/-1

                // read features
                for(int j = 0; j < 30; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
                //System.out.println(attributes[i][1][0]);

            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]< 0 ? 0 : 1));
        }

        return instances;
    }
}
