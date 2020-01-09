# Machine-Learning

### P1. Supervised learning
Select two interesting datasets and implemenet following algorithms
1. Decision trees with some form of pruning
2. Neural networks
3. Boosting
4. Support Vector Machines
5. k-nearest neighbors

Compare and contrast the different algorithms. What sort of changes might you make to each of those algorithms to improve performance? How fast were they in terms of wall clock time? Iterations? How was under or overfitting addressed


### P2. Random Optimization
Implement following four local random search algorithms. 
1. randomized hill climbing
2. simulated annealing
3. genetic algorithm
4. MIMIC

Use the first three algorithms to find good weights for a neural network. In particular, use them instead of backprop for the neural network built in assignment #1.

Create three optimization problem domains and apply all four search techniques. The first problem should highlight advantages of  genetic algorithm, the second of simulated annealing, and the third of MIMIC. Selected problem must illustrate relative strengths of each algorithm rather neatly.


### P3. Unsupervised Learning 
Implement following Unsupervised algorithm and perform comparative analysis.
1. k-means clustering
2. Expectation Maximization

Next use following dimensionaltiy reduction techniques and rerun the analysis
1. PCA
2. ICA
3. Randomized Projections
4. Any other feature selection algorithm you desire


### P4. Reinforcement Learning

1. Design two interesting MDPs- one with "small" number of states, and the other has a "large" number of states.
2. Solve each MDP using value iteration as well as policy iteration. 
3. Now pick your favorite reinforcement learning algorithm and use it to solve the two MDPs. 

Analysis - How many iterations does it take to converge? Which one converges faster? Why? Do they converge to the same answer? How did the number of states affect things, if at all?

How does RL perform, especially in comparison to the cases above where you knew the model, rewards, and so on? What exploration strategies did you choose? Did some work better than others?


