#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:39:28 2020

@author: clark
"""

"""
We'll try to write a metropolis coupled markov chain monte carlo
to solve the rotation map for a k-regular graph
"""

import numpy as np
from numpy import array as arr
from numpy.random import choice, rand
from numpy import matmul as mm

def fitness(expander):
    correct = mm(np.arange(1,expander.shape[0] + 1).reshape(expander.shape[0],1),
                 np.ones((1,expander.shape[1])))
    sorted_map = np.sort(expander, axis = 0)
    fit = sum(sum(np.abs(sorted_map - correct)))
    return fit

#normalizing_coefficient = fitness(np.sort(expander, axis = 1))
def compute_normalizing_coefficient(expander):
    nc = fitness(np.sort(expander, axis = 1))
    return nc


def transpose(vec, n):
    v = vec.copy()
    if n < len(vec)-1:
        transpose = np.arange(n,n+2)
        v[transpose] = v[transpose[::-1]]
    else:
        v[arr([0,-1])] = v[arr([-1,0])]
    return v   

def mix_columns(expander, num_mixes = "all"):
    r,c = expander.shape
    temp = expander.copy()
    if num_mixes == "all":
        for k in range(r):
            rearrange = np.random.choice(np.arange(c),c,replace = False)
            temp[k,:] = temp[k,rearrange]
    else:
       N = np.min([r, num_mixes])
       row_mix = np.random.choice(np.arange(N), N, replace = False)
       for k in range(N):
           rearrange = np.random.choice(np.arange(c),c,replace = False)
           temp[row_mix[k],:] = temp[row_mix[k], rearrange]
        
    return temp

"""
Some of this code is pulled from Liz Slander's Git at
https://github.com/elsander/GoodEnoughAlgs/blob/master/MCMCMC.py


"""

def InitializeTemps(T_min, T_max, num_chains):
    # Uniform spacing here, but this can be done differently
    # Log spacing is another option. In my experience this
    # has led to overall higher swap acceptance rates.
    T_step = (T_max - T_min) / (num_chains - 1)
    Ts = [T_max - n * T_step for n in range(num_chains)]
    return Ts

def InitializeChains(expander, T_min, T_max, num_chains):
    r,c = expander.shape
    if num_chains < 5:
        num_chains = 5
    Ts = InitializeTemps(T_min, T_max, num_chains)    
    expander0 = np.sort(expander, axis = 1)
    row_sums = np.sum(expander0, axis = 1)
    big_rows = np.argsort(row_sums)[-5:]
    expander1 = expander0.copy()
    expander1[big_rows,:] = expander1[big_rows,::-1] #Reverse the five biggest row sums
    chains = [expander0, expander1]
    for k in range(num_chains - 2):
        chains.append(mix_columns(expander0.copy(),"all"))
    
    """
    We will put the chains in order of fitness.  We're looking for the 
    smallest, and so we organize the temperatures in ascending order
    so that the lowest fitness is the coldest chain.
    """
    
    fits = [fitness(chains[n]) for n in range(num_chains)]
    chain_order = np.argsort(fits)
    chains = [chains[chain_order[n]] for n in range(num_chains)]
    Ts = [Ts[chain_order[n]] for n in range(num_chains)]
    return chains, Ts


def flip_chains(chains, Temps, loc):
    if loc < len(Temps) - 1:
        chains[loc], chains[loc + 1] = chains[loc + 1], chains[loc]
        Temps[loc], Temps[loc + 1] = Temps[loc + 1], Temps[loc]
    return chains, Temps

def acceptance_probability_two_chains(coldFit, hotFit, coldTemp, hotTemp, normalizing_coefficient):
    # the fitness of the original exapnder map is 646, 
    # so we're going to normalize by that
    nc = normalizing_coefficient
    ap = np.min([np.exp( ((coldFit - hotFit)/nc) * (1 / coldTemp - 1 / hotTemp)), 2])
    return ap
    
def acceptance_probability_one_chain(oldFit, newFit, Temp, normalizing_coefficient):
    # the fitness of the original exapnder map is 646, 
    # so we're going to normalize by that
    nc = normalizing_coefficient
    ap = np.exp((oldFit - newFit)/(nc*Temp))
    return ap

def neighbor(expander, max_iterations):
     for its in range(max_iterations):
         temp = expander.copy()
         for k in range(expander.shape[0]):
            temp_row = temp[k,:]
            loc = np.random.randint(0,expander.shape[1])
            temp[k,:] = transpose(temp_row, loc)
         if fitness(temp) < fitness(expander):
             expander = temp
     return temp
    
def MC3(expander, T_min, T_max, num_chains, n_steps_total = 1000):
    fit_history = []
    n_steps_swap = 20
    nc = compute_normalizing_coefficient(expander)
    """
    This is the number of steps we'll allow each chain to evolve
    before checking to switch chains
    """
    chains, Temps = InitializeChains(expander, T_min, T_max, num_chains)
    for N in range(n_steps_total):
        #print(N)
        if N % n_steps_swap == 0:
            for k in range(len(chains) - 1):
                 ap2 = acceptance_probability_two_chains(fitness(chains[k]), fitness(chains[k+1]), 
                                                        Temps[k], Temps[k+1], nc) 
                 if ap2 > rand():
                     flip_chains(chains, Temps, loc = k)
                     #print("Flipping")
        fits = [fitness(chains[c]) for c in range(len(Temps))]             
        bestFit = np.min(fits)
        bestSol = chains[np.argmin(fits)]
        print(bestFit)
            
        for n in range(num_chains):
            #fits = [fitness(chains[c]) for c in range(len(Temps))]
            #bestFit = np.min(fits)
            #bestSol = chains[np.argmin(fits)]
            #print(bestFit)
            trialSol = neighbor(chains[n], 2000)
            trialFit = fitness(trialSol)
            ap1 = acceptance_probability_one_chain(fits[n], trialFit, Temps[n], nc) 
                
            if ap1 > rand():
                fits[n] = trialFit
                chains[n] = trialSol
                # update best solution if appropriate
             
                

        if N%10 == 0:
            #print(N)
            print(bestFit)
            fit_history.append(bestFit)

    return bestSol, fit_history    
    
 
"""
There is something terribly wrong in the logic here, I'm getting
worse and worse "best Fits"

Will try again tomorrow
"""    
