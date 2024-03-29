# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 12:18:27 2018

@author: ac104q
GW samples

"""
#import numpy as np

maze =  [[0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 1, 0, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 1, 1, 1, 1, 0],
                 [0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0]]

# 15x15
ac_thunt_15 = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
          
# 13x13
ac_thunt =[[0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,1,0,0,0,0,0,0,0,0,0,0],
			[0,0,1,0,0,0,-4,-4,-4,-4,-4,0,0],
			[0,0,1,0,0,0,0,0,0,0,-4,0,0],
			[0,0,1,0,0,0,0,0,0,0,-4,0,0],
			[0,0,1,0,0,0,0,0,0,0,-4,0,0],
			[0,0,1,0,0,0,0,0,0,0,-4,0,0],
			[0,0,1,0,0,0,0,0,0,0,-4,0,0],
			[0,0,1,1,1,1,1,1,1,1, 0,0,0],
			[0,0,0,0,0,0,0,0,0,0, 0,0,0],
			[0,0,0,0,0,0,0,0,0,0, 0,0,0],
			[0,0,0,0,0,0,0,0,0,0, 0,0,0] ]

num=-4
#ac_thunt[(7,8)] = -2  
#ac_thunt[(6,6)] = -2  
#ac_thunt[(3,12)] = num
#ac_thunt[(4,12)] = num 
#ac_thunt[(5,12)] = num
#ac_thunt[(6,12)] = num 
#ac_thunt[(7,12)] = num 
#ac_thunt[(8,12)] = num 
#ac_thunt[(9,12)] = num 
#ac_thunt[(3,11)] = num 
#ac_thunt[(3,10)] = num 
#ac_thunt[(3,9)] = num 
#ac_thunt[(3,8)] = num 

ac_thunt_2 = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,-4,-4,-4,-4,-4,-4,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,-4,0,0],
    [0,0,1,1,1,1,1,1,0,0,-4,0,0],
    [0,0,1,0,0,0,0,0,0,0,-4,0,0],
    [0,0,1,0,0,1,0,0,0,0,-4,0,0],
    [0,0,1,1,1,1,0,0,0,0,-4,0,0],
    [0,0,1,0,0,0,0,0,0,0,-4,0,0],
    [0,0,1,0,0,0,0,0,0,-4,-4,0,0],
    [0,1,1,0,0,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0]]

ac_thunt_3 = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,-40,-40,-40,-40,-40,-40,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,-40,0,0],
    [0,0,1,1,1,1,1,1,0,0,-40,0,0],
    [0,0,1,0,0,0,0,0,0,0,-40,0,0],
    [0,0,1,0,0,1,0,0,0,0,-40,0,0],
    [0,0,1,1,1,1,0,0,0,0,-40,0,0],
    [0,0,1,0,0,0,0,0,0,0,-40,0,0],
    [0,0,1,0,0,0,0,0,0,-40,-40,0,0],
    [0,1,1,0,0,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0]]

ac_thunt_4 = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,40,40,40,40,40,40,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,40,0,0],
    [0,0,1,1,1,1,1,1,0,0,40,0,0],
    [0,0,1,0,0,0,0,0,0,0,40,0,0],
    [0,0,1,0,0,1,0,0,0,0,40,0,0],
    [0,0,1,1,1,1,0,0,0,0,40,0,0],
    [0,0,1,0,0,0,0,0,0,0,40,0,0],
    [0,0,1,0,0,0,0,0,0,40,40,0,0],
    [0,1,1,0,0,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0]]



ac_lybrinth_30 = [[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
            [1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,0],
            [1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0],
            [1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0],
            [1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0],
            [1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0],
            [1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0],
            [1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
            [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0],
            [1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0],
            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0],
            [1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0],
            [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0],
            [1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1],
            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
            [1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1],
            [1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],
            [1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0],
            [1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0],
            [1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0],
            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0],
            [1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1],
            [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0],
            [1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1],
            [1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
            [1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0],
            [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
            [0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0]]

ac_lybrinth_20 = [
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
            [0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
            [1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0],
            [1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
            [1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0],
            [1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0],
            [1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1],
            [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],
            [1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0],
            [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],
            [1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0],
            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0],
            [1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
            [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
            [1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0],
            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0],
            [1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1],
            [1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],
            [1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0],
            [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0 ]]




#ac_lybrinth_30_v0 =[[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#            [0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],
#            [1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,0],
#            [1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0],
#            [1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0],
#            [1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0],
#            [1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0],
#            [1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0],
#            [1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1],
#            [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0],
#            [1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0],
#            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0],
#            [1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0],
#            [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0],
#            [1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1],
#            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
#            [1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1],
#            [1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],
#            [1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0],
#            [1,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0],
#            [1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0],
#            [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0],
#            [1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1],
#            [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0],
#            [1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1],
#            [1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
#            [1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1],
#            [1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0],
#            [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
#            [1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0]]


