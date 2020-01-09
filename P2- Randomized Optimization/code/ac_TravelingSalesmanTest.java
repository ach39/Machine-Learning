package opt.test;

import java.util.Arrays;
import java.util.Random;
import java.text.DecimalFormat;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;


/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ac_TravelingSalesmanTest {
    /** The n value */
    private static final int N = 250;
    /**
     * The test main
     * @param args ignored
     */
	private static DecimalFormat df = new DecimalFormat("0.000");
	private static DecimalFormat df5 = new DecimalFormat("0.00000");

    public static void main(String[] args) {
        System.out.println("N=" + N + ", Algo , iter , result , train_time, Run" );
        for(int k=0; k<3; k++){
            my_test(k+1);
        }

    }
    public static void my_test(int runId) {
		
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
		
		double result, train_time,start=0,end=0;
		
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        int [] ITERATIONS = { 100 ,500, 1000, 2500, 5000, 7500 }; //10000

/*

		for(int iter : ITERATIONS) {
			RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
			start = System.nanoTime();
			FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
			fit.train();
			result = ef.value(rhc.getOptimal());
			end = System.nanoTime();
			train_time = end-start;
			train_time /= Math.pow(10,9);
			//System.out.println(ef.value(rhc.getOptimal()));
			System.out.println("TSP,RHC, " + iter + "," + df5.format(result) + "," + df.format(train_time) + "," + runId);
		}
		//System.out.println(" ");

		double [] T = { 0.75 };
        for(int iter : ITERATIONS){
			for(double temp : T) {
			SimulatedAnnealing sa = new SimulatedAnnealing(1E12, temp , hcp);
			start = System.nanoTime();
			FixedIterationTrainer fit = new FixedIterationTrainer(sa, iter);
			fit.train();
			result = ef.value(sa.getOptimal());	
			end = System.nanoTime();
			train_time = end-start;
			train_time /= Math.pow(10,9);
			//System.out.println("TSP,SA, " + iter + ","+ temp + "," + result + "," + df.format(train_time));
			System.out.println("TSP,SA, " + iter + "," + df5.format(result) + "," + df5.format(train_time)+ "," + runId);
			}
		}
		//System.out.println(" ");
	*/
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(500, 300, 25, gap);
		for(int iter : ITERATIONS){
			start  = System.nanoTime();
			FixedIterationTrainer fit = new FixedIterationTrainer(ga, iter);
			fit.train();
			end = System.nanoTime();
			train_time = end-start;
			train_time /= Math.pow(10,9);
			result = ef.value(ga.getOptimal());
			System.out.println("TSP,GA," + iter + ","  + df5.format(result) + "," + df.format(train_time)+ "," + runId);
		}
		//System.out.println(" ");
    /*
        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
		for(int iter : ITERATIONS){
			MIMIC mimic = new MIMIC(200, 100, pop);
			start= System.nanoTime();
			FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iter);
			fit.train();
			result = ef.value(mimic.getOptimal());
			end = System.nanoTime();
			train_time = end-start;
			train_time /= Math.pow(10,9);
			System.out.println("TSP,MIMIC," + iter + "," + df5.format(result) + "," + df5.format(train_time)+ "," + runId);
		}
     */

/*
        for(int i=0; i < 1 ; i++){
            String algo = "GA";

            if(algo == "SA") {
                System.out.println("TSP,SA ,iter, temp , Result , train_time" );
                explore_sa(hcp,ef, i+1); }

            if(algo == "GA") {
                System.out.println("TSP,GA , config, iter, Result , train_time" );
                explore_ga(gap,ef, i+1); }

            if(algo == "MIMIC") {
                System.out.println("TSP,MIMIC ,config ,iter  Result , train_time" );
                explore_mimic(points, i+1); }
        }

*/



    } //main


    private static void explore_sa(HillClimbingProblem hcp,TravelingSalesmanEvaluationFunction ef, int run){
        int [] ITER = { 10, 100 ,500, 1000, 2500, 5000, 8000,10000 };
        double [] T = {0.15 , 0.30, 0.45, 0.60, 0.75,0.95 };
        double start,end,train_time,result ;
        for(int iter : ITER){
            for(double temp : T) {
                SimulatedAnnealing sa = new SimulatedAnnealing(1E12, temp , hcp);
                start = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(sa, iter);
                fit.train();
                result = ef.value(sa.getOptimal());
                end = System.nanoTime();
                train_time = end-start;
                train_time /= Math.pow(10,9);
                System.out.println("TSP,SA, " + iter + ","+ temp + "," + df5.format(result) + "," + df.format(train_time) + "," + run );
            }
        }
    } //explore_sa

    private static void explore_ga(GeneticAlgorithmProblem gap,TravelingSalesmanEvaluationFunction ef, int run) {
        int[] pop_sz = {500, 5000};
        //int[] mate   = {5, 10, 25, 50, 120, 300};
        //int[] mutate = {2, 5, 8, 10, 20, 25};

        int [] ITER = { 100 ,500, 1000, 2500, 5000 };
        double start,end,train_time,result ;
        for(int iter : ITER){
            for (int i = 0; i < pop_sz.length; i++) {
                //StandardGeneticAlgorithm ga_2 = new StandardGeneticAlgorithm(pop_sz[i], mate[i], mutate[i], gap);
                StandardGeneticAlgorithm ga_2 = new StandardGeneticAlgorithm(pop_sz[i], (int)(pop_sz[i]*0.6), (int)(pop_sz[i]*0.05), gap);
                start = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(ga_2, iter);
                fit.train();
                end = System.nanoTime();
                train_time = end - start;
                train_time /= Math.pow(10, 9);
                result = ef.value(ga_2.getOptimal());
                System.out.println("TSP,GA-2,[" + pop_sz[i] + "-" + (int)(pop_sz[i]*0.6) + "-" + (int)(pop_sz[i]*0.05) + "]," + iter + ","
                        + df5.format(result) + "," + df.format(train_time) +","+run);
            }
        }
    }//explore ga


    private static void explore_ga_2(GeneticAlgorithmProblem gap,TravelingSalesmanEvaluationFunction ef, int run) {
        int pop_sz = 500;
        int[] mutate = {1, 3, 5, 7, 10 };


        int [] ITER = { 100 ,500, 1000, 2500, 5000, 8000 };
        double start,end,train_time,result ;
        for(int iter : ITER){
            for (int i = 0; i < mutate.length; i++) {
                //StandardGeneticAlgorithm ga_2 = new StandardGeneticAlgorithm(pop_sz[i], mate[i], mutate[i], gap);
                StandardGeneticAlgorithm ga_2 = new StandardGeneticAlgorithm(pop_sz, (int)(pop_sz*0.6), (int)(pop_sz*mutate[i]/100), gap);
                start = System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(ga_2, iter);
                fit.train();
                end = System.nanoTime();
                train_time = end - start;
                train_time /= Math.pow(10, 9);
                result = ef.value(ga_2.getOptimal());
                System.out.println("TSP,GA-2,[" + pop_sz + "-" +  (int)(pop_sz*0.6) + "-]," + (int)(pop_sz*mutate[i]/100) + "," + iter + ","
                        + df5.format(result) + "," + df.format(train_time) +","+run);
            }
        }
    }//explore ga


    private static void explore_mimic(double[][] points, int run) {
        // for mimic we use a sort encoding
        TravelingSalesmanSortEvaluationFunction ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        DiscreteUniformDistribution odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double start,end,train_time,result ;
        int[] samples = {10, 20, 50, 100, 200, 500};
        int[] to_keep = {5,  10, 25, 50,  120, 300};

        int [] ITER = { 10, 100 }; //,500, 1000, 2500, 5000, 8000 };

        for(int iter : ITER){
            for(int i=0; i<samples.length; i++){
                MIMIC mimic = new MIMIC(samples[i], to_keep[i], pop);
                start= System.nanoTime();
                FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iter);
                fit.train();
                result = ef.value(mimic.getOptimal());
                end = System.nanoTime();
                train_time = end-start;
                train_time /= Math.pow(10,9);
                System.out.println("TSP,MIMIC,[" + samples[i]+ "-"+ to_keep[i] + "]," + iter + "," + df5.format(result) + "," + df5.format(train_time));

            }

        }

    } //explore_mimic






}//ac_TravelingSalesmanTest






