#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:30:24 2020

@author: clark
"""


import numpy as np
from numpy import array as arr
from numpy.random import choice, rand
"""
expander = arr([[ 6, 12, 11,  5],
       [17, 11, 18, 12],
       [18,  6, 17,  5],
       [15,  7,  9, 13],
       [ 1, 15, 13,  3],
       [ 9,  3,  7,  1],
       [ 6, 10, 12,  4],
       [12, 10, 16, 18],
       [ 6, 16, 18,  4],
       [13,  7, 14,  8],
       [14,  2,  1, 13],
       [ 2,  7,  1,  8],
       [ 5, 10,  4, 11],
       [16, 17, 10, 11],
       [ 5, 16,  4, 17],
       [ 8,  9, 14, 15],
       [ 2, 14, 15,  3],
       [ 3,  8,  2,  9]])
"""
correct = np.arange(1,19).reshape(18,1)@np.ones((1,4))


def score(expander):
    correct = np.arange(1,expander.shape[0]+1).reshape(expander.shape[0],1)@np.ones((1,expander.shape[1]))
    sorted_map = np.sort(expander, axis = 0)
    score = sum(sum(np.abs(sorted_map - correct)))
    return score
    

"""
Write an MCMC to try and solve this
"""
def expander_switcher(expander, max_iterations):
    count = 0
    best_score = score(expander)
    best_expander = expander.copy()
    correct = np.arange(1,expander.shape[0]+1).reshape(expander.shape[0],1)@np.ones((1,expander.shape[1]))
    
    while (count < max_iterations) and (sum(sum(np.abs(np.sort(expander, axis = 0) - correct ))) > 0):
        temp_score = sum(sum(np.abs(np.sort(expander,axis=0) - correct)))
        temp_expander = expander.copy()
        for N in range(expander.shape[0]):
            r = choice(np.arange(expander.shape[1]),2, replace = False)
            temp_expander[N,r] = temp_expander[N,r[::-1]]
            new_score = sum(sum(np.abs(np.sort(temp_expander, axis=0) - correct)))
        if new_score < temp_score:
            expander = temp_expander
            count += 1
        else:
            if (0.1+temp_score)/(0.1+new_score) > rand():
                expander = temp_expander
                if count%1000 == 0:
                    print("jumped")
                    #count += 1
        if new_score < best_score:
            best_score = new_score
            best_expander = temp_expander
            print(best_score)
            
    return best_expander    
#%%            
def expander_minimizer(expander_map):
    expander = expander_map.copy()
    column_sums = sum(expander)
    row_sums = sum(expander.T)
    row_order = np.argsort(row_sums)
    rmax = np.argmax(column_sums)
    rmin = np.argmin(column_sums)
    half_set = expander.shape[0]//2
    #print(rmax,rmin)
    which_half = choice([-1,1])
    for n in range(half_set):
        temp_row = expander[which_half*row_order[n]]
        #print(temp_row)
        temp_max = np.argmax(temp_row)
        temp_min = np.argmin(temp_row)
        expander[which_half*row_order[n],arr([temp_max,temp_min])] = expander[which_half*row_order[n],arr([temp_min,temp_max])] 
    for m in range(half_set):
        #temp_row2 = expander[-row_order[m]]
        #print(temp_row2)
        expander[-which_half*row_order[m],arr([int(rmax),int(rmin)])]= expander[-which_half*row_order[m],arr([int(rmin),int(rmax)])]
    return expander


#%%


def transpose(vec, n):
    v = vec.copy()
    if n < len(vec)-1:
        transpose = np.arange(n,n+2)
        v[transpose] = v[transpose[::-1]]
    else:
        v[arr([0,-1])] = v[arr([-1,0])]
    return v   
    
def expander_transposer(expander, max_iterations = 100000):
    counter = 0
    best_score = 1000
    while (score(expander) > 1) and (counter < max_iterations):
        for k in range(expander.shape[0]):
            temp = expander.copy()
            temp_row = temp[k,:]
            #print(temp_row)
            loc = np.random.randint(0,expander.shape[1])
            temp[k,:] = transpose(temp_row, loc)
            #print(score(temp))
            if score(temp) < score(expander):
                expander = temp
                #print(score(temp))
            elif score(expander)/score(temp) > (9+rand())/10:
                expander = temp
            if score(temp) < best_score:
                best_score = score(temp)
                print("The new best score is {0}".format(best_score))
        counter += 1
        if score(expander) == 0:
            print("solved!")
            print(expander)
    return expander#, counter, score(temp)
        
        
    
