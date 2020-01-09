# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:57:10 2017

@author: Jon
"""

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from ac_GW_samples import *

import matplotlib
matplotlib.rcParams.update({'font.size': 10})
#matplotlib.rcParams['figure.titlesize'] = 'large'
matplotlib.rcParams['figure.figsize'] = 6,6

#infile = './Policy Easy Iter 43 Policy Map.pkl'

def plot_it(infile, title=''):
    with open(infile,'rb') as f:
        #arr = pkl.load(f, encoding='latin1')
        arr = pkl.load(f)
    
    lookup = {'None': (0,0),
              '>': (1,0),
            'v': (0,-1),
            '^':(0,1),
            '<':(-1,0)}    
    
    n= len(arr)
    arr = np.array(arr)    
    X, Y = np.meshgrid(range(1,n+1), range(1,n+1))    
    U = X.copy()
    V = Y.copy()
    for i in range(n):
        for j in range(n):
            U[i,j]=lookup[arr[n-i-1,j]][0]
            V[i,j]=lookup[arr[n-i-1,j]][1]
    
    #plt.figure(figsize=(10, 10))
    plt.figure()
    title_obj = plt.title(title)
    plt.setp(title_obj, color='b')
    plt.title(title)
    #plt.title('Arrows scale with plot width, not view')
    Q=plt.quiver(X, Y, U, V,headaxislength=5,pivot='mid',angles='xy',scale_units='xy',scale=1 )
    #Q=plt.quiver(X, Y, U, V, headaxislength=10, pivot='mid',scale_units='xy',scale=1 )
    #Q=plt.quiver(X, Y, U, V, color='g',linewidths = 20, headaxislength=10,scale_units='xy')  
    
    plt.xlim((0,n+1))
    plt.ylim((0,n+1))
    plt.tight_layout()


def plot_TH() :
    to_print=[1,10,20,60]
    out = 'TH_Logs/VI/'
    for i in to_print :
        world = 'tHunt'
        #f= out + 'Value {} Iter {} Policy Map.pkl'.format(world,i)
        f= out + 'VI {} Iter {} D-0.99 Policy Map.pkl'.format(world,i)
        plot_it(f ,'TreasureHunt: VI  (' + str(i) + ')')
#plot_TH()

def plot_lyb() :
    to_print=[1,73]
    out = 'LYB_Logs/VI/'
    for i in to_print :
        world = 'lybrinth'
        #f= out + 'Value {} Iter {} Policy Map.pkl'.format(world,i)
        f= out + 'VI {} Iter {} D-0.99 Policy Map.pkl'.format(world,i)
        plot_it(f ,'Lybrinth : VI (' + str(i) + ')')

#plot_lyb()
        
        
#########################################################################    

#Easy_JT = [
#      [ 0, 0, 0, -1, 0, 0, 0, 0,-5,0],
#      [ 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
#      [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
#      [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
#      [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
#      [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
#      [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
#      [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
#      [ 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
#      [ 0,-3, 0, 0, 0, 0, 0, 0, 0, 0],              
#      ]
    
def print_grid(userMap) :

    fig, ax = plt.subplots()
    ax.matshow(userMap, interpolation='nearest')
    #ax.matshow(userMap, interpolation='nearest',cmap='seismic')
    #ax.matshow(userMap, interpolation='nearest',cmap=plt.cm.gray)
    plt.cm.gray
    for (i, j), z in np.ndenumerate(userMap):
        if z!=0 :
            ax.text(j, i, '{}'.format(z), ha='center', va='center')
    plt.show()
    
    
#print_grid(Easy_JT)
#print_grid(Hard_JT)
    
#print_grid(Easy_YC)
#print_grid(Hard_YC)

#print_grid(ac_thunt_15)    
#print_grid(ac_thunt_2)


print_grid(ac_lybrinth_30)

#print_grid(maze)