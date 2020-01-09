package opt.test;

import java.util.Arrays;
import java.util.Random;

import opt.ga.MaxKColorFitnessFunction;
import opt.ga.Vertex;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author kmandal
 * @version 1.0
 */
public class ac_MaxKColoringTest {
    /** The n value */
    //private static final int N = 800; 	// number of vertices
    private static final int L = 4; 	// L adjacent nodes per vertex
    private static final int K = 6; 	// K possible colors


    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {

        System.out.println("AdjNodes=" + (int) L + "  colors=" + (int) K);
        System.out.println("vertices , Algo, Opt_Val ,status , time ,iter ");

        int[] N_range = {1000}; //{ 200, 400, 600, 800,1000, 1200 };

        for (int N: N_range){

            Random random = new Random(N * L);
            // create the random velocity
            Vertex[] vertices = new Vertex[N];
            for (int i = 0; i < N; i++) {
                Vertex vertex = new Vertex();
                vertices[i] = vertex;
                vertex.setAdjMatrixSize(L);
                for (int j = 0; j < L; j++) {
                    vertex.getAadjacencyColorMatrix().add(random.nextInt(N * L));
                }
            }


            /*
            for (int i = 0; i < N; i++) {
                Vertex vertex = vertices[i];
                System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
            }*/
            // for rhc, sa, and ga we use a permutation based encoding
            MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
            Distribution odd = new DiscretePermutationDistribution(K);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new SingleCrossOver();
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            Distribution df = new DiscreteDependencyTree(.1);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            long starttime = System.currentTimeMillis();


            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
            fit.train();
 /*
            System.out.println( N + ", RHC, " + ef.value(rhc.getOptimal()) + ", " + ef.foundConflict() + ", " + (System.currentTimeMillis() - starttime));
    //        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
    //        System.out.println(ef.foundConflict());
    //        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
    //        System.out.println("============================");


            starttime = System.currentTimeMillis();
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .1, hcp);
            fit = new FixedIterationTrainer(sa, 20000);
            fit.train();
            System.out.println(N + ", SA, " + ef.value(sa.getOptimal()) + ", " + ef.foundConflict() + ", " + (System.currentTimeMillis() - starttime));
    //        System.out.println("SA: " + ef.value(sa.getOptimal()));
    //        System.out.println(ef.foundConflict());
    //        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
    //        System.out.println("============================");
*/
            int [] ITERATIONS = { 1,2,3,4,5,6,7,8,9,10} ;
            for( int iter : ITERATIONS ) {
                starttime = System.currentTimeMillis();
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 60, 10, gap);
                //StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(150, 95, 25, gap);
                fit = new FixedIterationTrainer(ga, iter);
                fit.train();
                //System.out.println(N + ", GA, " + ef.value(ga.getOptimal()) + ", " + ef.foundConflict() + ", " + (System.currentTimeMillis() - starttime) +","+ iter + "," + ef.feval);
                System.out.println(N + ", GA, " + ef.value(ga.getOptimal()) + ", " + ef.foundConflict() + ", " + (System.currentTimeMillis() - starttime) +","+ iter );
                //        System.out.println("GA: " + ef.value(ga.getOptimal()));
                //        System.out.println(ef.foundConflict());
                //        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
                //        System.out.println("============================");


                starttime = System.currentTimeMillis();
                MIMIC mimic = new MIMIC(100, 50, pop);
                fit = new FixedIterationTrainer(mimic, iter);
                fit.train();
                //System.out.println(N + ", MIMIC, " + ef.value(mimic.getOptimal()) + ", " + ef.foundConflict() + ", " + (System.currentTimeMillis() - starttime) +","+ iter + ","+ ef.feval);
                System.out.println(N + ", MIMIC, " + ef.value(mimic.getOptimal()) + ", " + ef.foundConflict() + ", " + (System.currentTimeMillis() - starttime) +","+ iter );
                //        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
                //        System.out.println(ef.foundConflict());
                //        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));

            }

        }//for N_range
    } //main
} //class
