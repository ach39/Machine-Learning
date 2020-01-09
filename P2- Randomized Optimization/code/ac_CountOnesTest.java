package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
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
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ac_CountOnesTest {
    /** The n value */
    private static final int [] N = {100,200,300,400,500, 600, 800,1000};
    
    public static void main(String[] args) {
        for (int n : N ) {

            int[] ranges = new int[n];
            Arrays.fill(ranges, 2);
            //EvaluationFunction ef = new CountOnesEvaluationFunction();
            CountOnesEvaluationFunction ef = new CountOnesEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);


            int iter = 6000;
            long st=0, ms=0 ;
            double val=0 ;


            st = System.currentTimeMillis();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter);
            fit.train();
            ms = (System.currentTimeMillis() - st);
            val = ef.value(rhc.getOptimal());
            System.out.println(n + ", RHC, " + val + ", " + ms +","+ iter );
            //System.out.println(n + ", RHC, " + val + ", " + ms +","+ iter + "," + ef.feval);


            st = System.currentTimeMillis();
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, iter);
            fit.train();
            ms = (System.currentTimeMillis() - st);
            val = ef.value(sa.getOptimal());
            //System.out.println(n + ", SA, " + val + ", " + ms +","+ iter + "," + ef.feval);
            System.out.println(n + ", SA, " + val + ", " + ms +","+ iter );

            st = System.currentTimeMillis();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, iter);
            fit.train();
            ms = (System.currentTimeMillis() - st);
            val= ef.value(ga.getOptimal());
            //System.out.println(n + ", GA, " + val + ", " + ms +","+ iter + "," + ef.feval);
            System.out.println(n + ", GA, " + val + ", " + ms +","+ iter );

            st = System.currentTimeMillis();
            MIMIC mimic = new MIMIC(50, 10, pop);
            fit = new FixedIterationTrainer(mimic, 100);
            fit.train();
            ms = (System.currentTimeMillis() - st);
            val = ef.value(mimic.getOptimal());
            //System.out.println(n + ", MIMIC, " + val + ", " + ms + ", 2" + "," + ef.feval);
            System.out.println(n + ", GA, " + val + ", " + ms +","+ iter  );

        }

    }
}