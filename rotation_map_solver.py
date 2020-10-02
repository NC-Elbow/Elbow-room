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
    
def expander_transposer(expander, max_iterations = 10000):
    counter = 0
    best_score = 10000
    best_expander = expander.copy()
    while (score(expander) > 1) and (counter < max_iterations):
        for k in range(expander.shape[0]):
            temp = expander.copy()
            temp_row = temp[k,:]
            #print(temp_row)
            loc = np.random.randint(0,expander.shape[1])
            temp[k,:] = transpose(temp_row, loc)
            #print(score(temp))
            temp_score = score(temp)
            if temp_score < score(expander):
                expander = temp
                #print(score(temp))
            elif score(expander)/temp_score > (9+rand())/10:
                expander = temp
            if temp_score < best_score:
                best_score = temp_score
                best_expander = temp
                if best_score < 100:
                    print("The new best score is {0}".format(best_score))
        counter += 1
        if score(expander) == 0:
            print("solved!")
            print(expander)
    return best_expander#, counter, score(temp)
        
        
"""
The new best score is 0.0
solved!
[[ 5 12  6 11]
 [17 18 11 12]
 [ 6  5 18 17]
 [15  7  9 13]
 [13 15  1  3]
 [ 1  3  7  9]
 [12  4 10  6]
 [18 16 12 10]
 [ 4  6 16 18]
 [14 13  8  7]
 [ 2  1 13 14]
 [ 7  8  2  1]
 [11 10  4  5]
 [10 11 17 16]
 [16 17  5  4]
 [ 9 14 15  8]
 [ 3  2 14 15]
 [ 8  9  3  2]]    


 for k in range(1000):
    if score(new_exp) == 0:
        break
    else:    
        new_exp = expander_transposer(np.sort(new_exp, axis = 1), 2500)
    print(k, score(new_exp))
"""


"""
Going to build up a 46 vertex, 5 regular graph and a newer monte carlo method 
for solving.

This is 4 copies of K_4,4 graphs with two left side pairs, and the 
right sides are connected to another vertex.  There are 5 vertices 
connected to a single 'central' vertex

Here's the map:
    
g46 = arr([[ 2., 21., 23., 24., 25.],
       [ 1., 21., 23., 24., 25.],
       [ 4., 21., 23., 24., 25.],
       [ 3., 21., 23., 24., 25.],
       [ 6., 25., 26., 27., 28.],
       [ 5., 25., 26., 27., 28.],
       [ 8., 25., 26., 27., 28.],
       [ 7., 25., 26., 27., 28.],
       [10., 29., 30., 31., 32.],
       [ 9., 29., 30., 31., 32.],
       [12., 29., 30., 31., 32.],
       [11., 29., 30., 31., 32.],
       [14., 33., 34., 35., 36.],
       [13., 33., 34., 35., 36.],
       [16., 33., 34., 35., 36.],
       [15., 33., 34., 35., 36.],
       [18., 37., 38., 39., 40.],
       [17., 37., 38., 39., 40.],
       [20., 37., 38., 39., 40.],
       [19., 37., 38., 39., 40.],
       [ 1.,  2.,  3.,  4., 41.],
       [ 1.,  2.,  3.,  4., 41.],
       [ 1.,  2.,  3.,  4., 41.],
       [ 1.,  2.,  3.,  4., 41.],
       [ 5.,  6.,  7.,  8., 42.],
       [ 5.,  6.,  7.,  8., 42.],
       [ 5.,  6.,  7.,  8., 42.],
       [ 9., 10., 11., 12., 43.],
       [ 9., 10., 11., 12., 43.],
       [ 9., 10., 11., 12., 43.],
       [ 9., 10., 11., 12., 43.],
       [13., 14., 15., 16., 44.],
       [13., 14., 15., 16., 44.],
       [13., 14., 15., 16., 44.],
       [13., 14., 15., 16., 44.],
       [17., 18., 19., 20., 45.],
       [17., 18., 19., 20., 45.],
       [17., 18., 19., 20., 45.],
       [17., 18., 19., 20., 45.],
       [21., 22., 23., 24., 46.],
       [25., 26., 27., 28., 46.],
       [29., 30., 31., 32., 46.],
       [33., 34., 35., 36., 46.],
       [37., 38., 39., 40., 46.],
       [41., 42., 43., 44., 45.]])



to mix up the rows
for k in range(expander.shape[0]):
    expander[k,:] = expander[k,choice(np.arange(expander.shape[1]), 
                                      expander.shape[1], replace = False)]
   

    

"""

def make_doubly_stochastic(matrix):
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    if nrows != ncols:
        print("Not a square matrix")
    else:
        while (sum(np.abs(sum(matrix)  - np.ones(nrows) ) > 0.00001 )) and (sum(np.abs(sum(matrix.T)  - np.ones(nrows) ) > 0.00001)):
            matrix = matrix/(sum(matrix))
            matrix = (matrix.T / (sum(matrix.T)))
    return matrix    


def mix_columns(expander, num_mixes = "all"):
    r,c = expander.shape
    if num_mixes == "all":
        for k in range(r):
            rearrange = np.random.choice(np.arange(c),c,replace = False)
            expander[k,:] = expander[k,rearrange]
    else:
       N = np.min([r, num_mixes])
       row_mix = np.random.choice(np.arange(N), N, replace = False)
       for k in range(N):
           rearrange = np.random.choice(np.arange(c),c,replace = False)
           expander[row_mix[k],:] = expander[row_mix[k], rearrange]
        
    return expander    
    
"""
Let's give ourselves a shot with the metropolis coupled markov chain monte carlo
(MCMCMC) method.
The difficulty here is to decide what the "distributions" should look like
and what a stationary distribution really means.
I suppose since there are multiple solutions, even multiple families
we can pick number of chains = regularity of graph.
A stationary state is a "solution."  

Now we need to decide how to switch between chains
and what are some good initializations.
"""

#def mc3Transposer(expander, nchains):
    
    