'''
Created on Mar 31 9, 2018
Adapted from J.Tay code
'''
import java
import csv
from collections import deque
from collections import defaultdict
from time import clock
import pickle
import sys
sys.path.append('./burlap.jar')

from burlap.behavior.learningrate import ExponentialDecayLR, SoftTimeInverseDecayLR
from burlap.behavior.policy import Policy;
from burlap.behavior.singleagent import EpisodeAnalysis;
from burlap.behavior.singleagent.auxiliary import StateReachability;
from burlap.behavior.singleagent.auxiliary.valuefunctionvis import ValueFunctionVisualizerGUI;
from burlap.behavior.singleagent.learning.tdmethods import QLearning;
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent import SADomain;
from burlap.oomdp.singleagent.environment import SimulatedEnvironment;
from burlap.oomdp.singleagent.explorer import VisualExplorer;
from burlap.oomdp.statehashing import HashableStateFactory;
from burlap.oomdp.statehashing import SimpleHashableStateFactory;
from burlap.oomdp.visualizer import Visualizer;
from burlap.behavior.valuefunction import ValueFunction;
from burlap.domain.singleagent.gridworld import GridWorldDomain;
from burlap.oomdp.core import Domain;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;

from burlap.assignment4 import BasicGridWorld;
from burlap.assignment4.util import MapPrinter;
from burlap.assignment4.util import BasicRewardFunction;
from burlap.assignment4.util import BasicTerminalFunction;
from burlap.assignment4.EasyGridWorldLauncher import visualizeInitialGridWorld
from burlap.assignment4.util.AnalysisRunner import calcRewardInEpisode, simpleValueFunctionVis,getAllStates

from ac_GW_samples import *



VI = True
PI = True
QL = True

if 0 :
    discount = [ 0.90]
    MAX_DELTA_BREAK = False
    to_print = [1]
else :    
    #discount = [0.45, 0.75, 0.85, 0.99]
    discount = [0.45, 0.75, 0.99]
    to_print = [1,40]
    MAX_DELTA_BREAK = True

def dumpCSV(nIter, times,rewards,steps,convergence,discount , method, world ,qval= None):
    
    iters = range(1,nIter+1)
    assert len(iters)== len(times)
    assert len(iters)== len(rewards)
    assert len(iters)== len(steps)
    assert len(iters)== len(convergence)
    
    disc    = [discount] *len(iters)
    name    = [method + ' D-' + str(discount) + ' ' + world] * len(iters)
    
    if qval :
        fname = 'LYB_logs/QL' + '/{} {} D-{}.csv'.format(world,method,discount)
        hdr = 'iter,time,reward,steps,convergence , lr, qInit,Epsilon,discount, method \n ' 
                        #str(qval[0]) + ',' +  str(qval[1]) + ',' + str(qval[2]) + ',' + world + 'GW,'+  method + '\n'
        lr      = [qval[0]] * len(iters)
        qInit   = [qval[1]] * len(iters)
        epsilon = [qval[2]] * len(iters)
                    
    else :
        fname = 'LYB_logs/' + method + '/{} {} D-{}.csv'.format(world,method,discount)
        hdr = 'iter,time,reward,steps,convergence ,discount' + ',' + world + 'GW,'+  method + '\n'
    
    
    print 'Writing to file :', fname
    with open(fname,'wb') as f:
        f.write(hdr)
        writer = csv.writer(f,delimiter=',')
        if qval :
            writer.writerows(zip(iters,times,rewards,steps,convergence,lr,qInit,epsilon,disc,name))
        else :
            writer.writerows(zip(iters,times,rewards,steps,convergence,disc,name))
    
    
def runEvals(initialState,plan,rewardL,stepL):
    r = []
    s = []
    for trial in range(evalTrials):
        ea = plan.evaluateBehavior(initialState, rf, tf,300);
        r.append(calcRewardInEpisode(ea))
        s.append(ea.numTimeSteps())
    rewardL.append(sum(r)/float(len(r)))
    stepL.append(sum(s)/float(len(s))) 


def comparePolicies(policy1,policy2):
    assert len(policy1)==len(policy1)
    diffs = 0
    for k in policy1.keys():
        if policy1[k] != policy2[k]:
            diffs +=1
    return diffs

def mapPicture(javaStrArr):
    out = []
    for row in javaStrArr:
        out.append([])
        for element in row:
            out[-1].append(str(element))
    return out

def dumpPolicyMap(javaStrArr,fname):
    fname = 'LYB_logs/' + fname
    pic = mapPicture(javaStrArr)
    with open(fname,'wb') as f:
        pickle.dump(pic,f)
    
if __name__ == '__main__':
    world = 'lybrinth'
    NUM_INTERVALS   = MAX_ITERATIONS = 200 ; 
    evalTrials      = 100;

    userMap = ac_lybrinth_20
    #userMap = ac_lybrinth_30
    
    n = len(userMap)
    tmp = java.lang.reflect.Array.newInstance(java.lang.Integer.TYPE,[n,n])
    for i in range(n):
        for j in range(n):
            tmp[i][j]= userMap[i][j]
    userMap = MapPrinter().mapToMatrix(tmp)
    maxX = maxY= n-1
    
    print '\n\n ====== > Calling Lybrinth 30x30 :  Max X, Y - ' , maxX , ' ' , maxY
     
    gen             = BasicGridWorld(userMap,maxX,maxY)
    domain          = gen.generateDomain()
    initialState    = gen.getExampleState(domain);

    rf  = BasicRewardFunction(maxX,maxY,userMap)
    tf  = BasicTerminalFunction(maxX,maxY)
    env = SimulatedEnvironment(domain, rf, tf,initialState);
    
#    Print the map that is being analyzed
    print "/////{} Grid World Analysis/////\n".format(world)
    MapPrinter().printMap(MapPrinter.matrixToMap(userMap));

    hashingFactory = SimpleHashableStateFactory()
    increment   = MAX_ITERATIONS/NUM_INTERVALS
    timing      = defaultdict(list)
    rewards     = defaultdict(list)
    steps       = defaultdict(list)
    convergence = defaultdict(list)
    allStates   = getAllStates(domain,rf,tf,initialState)
    
    #policy_converged    = defaultdict(list)
    #last_policy         = defaultdict(list)
    
    iterations = range(1, MAX_ITERATIONS+1)
          
    #####################  Value Iteration ########################################################
    if VI :
        for disc in discount :
            timing      = defaultdict(list)
            rewards     = defaultdict(list)
            steps       = defaultdict(list)
            convergence = defaultdict(list)
            
            vi = ValueIteration(domain, rf, tf, disc, hashingFactory,-1, 1);    
            vi.setDebugCode(-1) 
            vi.performReachabilityFrom(initialState)
            vi.toggleUseCachedTransitionDynamics(False)
            
            print "// {} GW Value Iteration Analysis  //".format(world)
            timing['Value'].append(0)    
            
            for nIter in iterations:      
                startTime = clock()  
                vi.runVI()
                timing['Value'].append(timing['Value'][-1]+clock()-startTime)
                p = vi.planFromState(initialState);        
                convergence['Value'].append(vi.latestDelta)           
                # evaluate the policy with evalTrials roll outs
                runEvals(initialState,p,rewards['Value'],steps['Value'])
                if nIter in to_print :
                    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'VI/VI {} Iter {} D-{} Policy Map.pkl'.format(world,nIter,disc))
                    #if nIter==1 : simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, "Value Iteration {}".format(nIter))
                    
                if vi.latestDelta <1e-6 and MAX_DELTA_BREAK :
                    print 'Treasure Hunt VI : Convergence at Iter= ' , nIter
                    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'VI/VI {} Iter {} D-{} Policy Map.pkl'.format(world,nIter,disc)) 
                    break
            #MapPrinter.printPolicyMap(vi.getAllStates(), p, gen.getMap());
            #simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, "Value Iteration{}".format(nIter))
            
            dumpCSV(nIter, timing['Value'][1:], rewards['Value'], steps['Value'],convergence['Value'],disc, 'VI', world )
      
      
    #####################  Policy Iteration ########################################################   
   
    if PI :
        for disc in discount: 
            timing      = defaultdict(list)
            rewards     = defaultdict(list)
            steps       = defaultdict(list)
            convergence = defaultdict(list)
            
            pi = PolicyIteration(domain, rf, tf, disc, hashingFactory, 1e-3, 10, 1)  
            pi.toggleUseCachedTransitionDynamics(False)   
            print "//{} Policy Iteration Analysis//".format(world)
            timing['Policy'].append(0)
            
            iterations = range(1, MAX_ITERATIONS+1)
            
            for nIter in iterations:
                startTime = clock()                         
                p = pi.planFromState(initialState);
                timing['Policy'].append(timing['Policy'][-1]+clock()-startTime)   
                policy = pi.getComputedPolicy()    
                current_policy = {state: policy.getAction(state).toString() for state in allStates} 
                convergence['Policy2'].append(pi.lastPIDelta)
                if nIter == 1:
                    convergence['Policy'].append(999)
                else:
                    convergence['Policy'].append(comparePolicies(last_policy,current_policy))       
                last_policy = current_policy                
                runEvals(initialState, p, rewards['Policy'], steps['Policy'])
                
                if nIter in to_print :
                    #if nIter==1 : simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration {}".format(nIter))
                    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'PI/PI {} Iter {} Policy Map.pkl'.format(world,nIter))
                
                if convergence['Policy2'][-1] <1e-6 and MAX_DELTA_BREAK:
                    print 'Treasure Hunt PI : Convergence at Iter= ' , nIter
                    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'PI/PI {} Iter {} Policy Map.pkl'.format(world,nIter))
                    break
            
            MapPrinter.printPolicyMap(pi.getAllStates(), p, gen.getMap());
            #simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration{}".format(nIter))
            print "\n\n\n"
            dumpCSV(nIter, timing['Policy'][1:], rewards['Policy'], steps['Policy'],convergence['Policy2'], disc, 'PI', world)
            
      
    #####################  Q-learning  ######################################################## 
    if QL :
        MAX_ITERATIONS = 300
        iterations = range(1,MAX_ITERATIONS+1)
    
        evalTrials = 10    # 100
 
        for lr in [0.1]:
            for qInit in [1]:  
                for epsilon in [0.7 ]:     
                     
#        for lr in [0.1,0.5, 0.9]:
#            for qInit in [0,50]:  
#                for epsilon in [0.2, 0.6, 0.9]:
                    for disc in discount:
                        print 'Init is ' , qInit
                        timing      = defaultdict(list)
                        rewards     = defaultdict(list)
                        steps       = defaultdict(list)
                        convergence = defaultdict(list)
                        
                        last10Rewards= deque([10]*10,maxlen=10)
                        Qname = 'QL L{:0.1f} q{:0.1f} E{:0.1f}'.format(lr,qInit,epsilon)
                        agent = QLearning(domain,disc,hashingFactory, qInit ,lr, epsilon)
                        agent.setDebugCode(0)
                        print "//Lybrinth {} Iteration Analysis//".format(Qname)
                        print Qname    
                        
                        for nIter in iterations:
                            #print " ====> Iter = ", nIter
                            startTime = clock()
                            if nIter%100==0:
                                print nIter
                            ea = agent.runLearningEpisode(env)
                            env.resetEnvironment()
                            agent.initializeForPlanning(rf, tf, 1)
                            p = agent.planFromState(initialState)     # run planning from our initial state
                            if len(timing[Qname])> 0:
                                timing[Qname].append(timing[Qname][-1]+clock()-startTime)   
                            else:
                                timing[Qname].append(clock()-startTime)     
                            #timing[Qname].append((clock()-startTime)*1000)
                            last10Rewards.append(agent.maxQChangeInLastEpisode)
                            convergence[Qname].append(sum(last10Rewards)/10.)
                            # evaluate the policy with one roll out visualize the trajectory
                            runEvals(initialState,p,rewards[Qname],steps[Qname])
                        
                        #MapPrinter.printPolicyMap(getAllStates(domain,rf,tf,initialState), p, gen.getMap());
                        print "\n\n"
                        #simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory, Qname+' {}'.format(nIter))
                        
                        #dumpCSV(iterations, timing[Qname], rewards[Qname], steps[Qname],convergence[Qname], world, Qname)
                        qval = [lr , qInit , epsilon]
                        dumpCSV(nIter, timing[Qname], rewards[Qname], steps[Qname],convergence[Qname],disc, Qname, world , qval)
    
    # ----------------------------------------------------------------------------------------------#
    if 0 :
    
        MAX_ITERATIONS  = NUM_INTERVALS = MAX_ITERATIONS*40;
        increment       = MAX_ITERATIONS/NUM_INTERVALS
        iterations      = range(1,MAX_ITERATIONS+1)
        
        for lr in [0.1]: 
            for qInit in [0]:  
                for epsilon in [0.9 ]:  
                    for disc in [0.75, 0.99 ]:
        
#        for lr in [0.1,0.5, 0.9]:
#            for qInit in [-50,0,50]:
#                for epsilon in [0.2, 0.6, 0.9]:
#                    for disc in discount:
                        
                        timing      = defaultdict(list)
                        rewards     = defaultdict(list)
                        steps       = defaultdict(list)
                        convergence = defaultdict(list)
      
                        last10Chg = deque([99]*10,maxlen=10)
                        Qname = 'QL L{:0.1f} q{:0.1f} E{:0.1f}'.format(lr,qInit,epsilon)
                        agent = QLearning(domain, disc ,hashingFactory, qInit, lr, epsilon,2000)
                        #agent.setLearningRateFunction(SoftTimeInverseDecayLR(1.,0.))
                        agent.setDebugCode(0)
                        print "//{} {} Iteration Analysis//".format(world,Qname)           
                        for nIter in iterations: 
                            if nIter % 200 == 0: print(nIter)			
                            startTime = clock()    
                            ea = agent.runLearningEpisode(env,2000)   
                            if len(timing[Qname])> 0:
                                timing[Qname].append(timing[Qname][-1]+clock()-startTime)   
                            else:
                                timing[Qname].append(clock()-startTime)             
                            env.resetEnvironment()
                            agent.initializeForPlanning(rf, tf, 1)
                            p = agent.planFromState(initialState)     # run planning from our initial state                
                            last10Chg.append(agent.maxQChangeInLastEpisode)
                            convergence[Qname].append(sum(last10Chg)/10.)          
                            # evaluate the policy with one roll out visualize the trajectory
                            runEvals(initialState,p,rewards[Qname],steps[Qname]) 
                            
                            if nIter == 600 or convergence[Qname][-1] <0.5 :
                                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()), \
                                  'QL/{} {} Iter {} Policy Map.pkl'.format(Qname,world,nIter))
                            if convergence[Qname][-1] < 0.5:
                                break
                        
                        print "\n\n\n"
                        qval = [lr , qInit , epsilon]
                        dumpCSV(nIter, timing[Qname], rewards[Qname], steps[Qname],convergence[Qname],disc, Qname, world , qval)
         
    
    
