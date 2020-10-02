#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:44:02 2020

@author: clark
"""

"""
The first attempt at MC^3 did not work properly.  I will try SAGA here.
Then I'll rewrite another MCMCMC trial and restrict worse solutions from 
happening
"""

"""
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

"""

import numpy as np
from numpy.random import rand, randint, choice
from numpy import array as arr
from numpy import matmul as mm

def fitness(expander):
    correct = mm(np.arange(1,expander.shape[0] + 1).reshape(expander.shape[0],1),
                 np.ones((1,expander.shape[1])))
    sorted_map = np.sort(expander, axis = 0)
    fit = sum(sum(np.abs(sorted_map - correct)))
    return fit

def fitness2(expander):
    correct = mm(np.arange(1,expander.shape[0] + 1).reshape(expander.shape[0],1),
                 np.ones((1,expander.shape[1])))
    sorted_map = np.sort(expander, axis = 0)
    fit = sum(sum(sorted_map != correct))
    return fit

#normalizing_coefficient = fitness(np.sort(expander, axis = 1))
def compute_normalizing_coefficient(expander):
    nc = fitness2(np.sort(expander, axis = 1))
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

def expander_transposer(expander, max_iterations = 10000):
    counter = 0
    best_score = 10000
    best_expander = expander.copy()
    while (fitness2(expander) > 1) and (counter < max_iterations):
        for k in range(expander.shape[0]):
            temp = expander.copy()
            temp_row = temp[k,:]
            #print(temp_row)
            loc = np.random.randint(0,expander.shape[1])
            temp[k,:] = transpose(temp_row, loc)
            #print(score(temp))
            temp_score = fitness2(temp)
            if temp_score < fitness2(expander):
                expander = temp
                #print(score(temp))
            elif fitness2(expander)/temp_score > (9+rand())/10:
                expander = temp
            if temp_score < best_score:
                best_score = temp_score
                best_expander = temp
                if best_score < 100:
                    print("The new best score is {0}".format(best_score))
        counter += 1
        if fitness2(expander) == 0:
            print("solved!")
            print(expander)
    return best_expander#, counter, score(temp)



def acceptance_probability(oldFit, newFit, Temp, normalizing_coefficient):
    ap = np.exp((oldFit - newFit)/(normalizing_coefficient * Temp))
    return ap

def neighbor(expander):
    # We'll transpose two elements from each row
    temp = expander.copy()
    for k in range(expander.shape[0]):
       temp_row = temp[k,:]
       loc = np.random.randint(0,expander.shape[1])
       temp[k,:] = transpose(temp_row, loc)
    return temp

def anneal(expander):
    T = 1
    T_min = 0.001
    cooling_rate = 0.8
    nc = fitness2(np.sort(expander.copy(), axis =1))
    # The above line prevents us from annealing and then annealing again
    # In order to mitigate this, we'll run a genetic algorithm
    best_fit = nc.copy()
    best_expander = expander.copy()
    old_sol = expander.copy()
    old_fit = nc.copy()
    num_searches_per_degree = 300
    while (T > T_min) and (best_fit > 0):
        for k in range(num_searches_per_degree):
            new_sol = neighbor(old_sol)
            new_fit = fitness2(new_sol)
            if (new_fit < old_fit):
                old_sol = new_sol
                old_fit = new_fit
            else:
                ap = acceptance_probability(old_fit, new_fit, T, nc)
                if (ap > rand()):
                    old_sol = new_sol
                    old_fit = new_fit
            if new_fit < best_fit:
                best_fit = new_fit.copy()
                best_expander = new_sol.copy()
                #print("Best fit is {0}".format(best_fit))
        T *= cooling_rate
    return best_expander, best_fit

"""
The annealer is working reasonably well.  
Now we will run it through a genetic algorithm.
The selection will be row by row.
mutation will be complete random reordering of a random number of rows.


in order to select rows randomly we need a selection of expander.shape[0] 0/1
in this case r,c = expander.shape
a = randint(0,2,r)
b = 1-a
the selection is then
a.reshape(r,1)*parent1 + b.reshape(r,1)*parent2
"""

def breed(expander1, expander2):
    r,c = expander1.shape
    # Make a random selection or rows from parent 1
    a = randint(0,2,size = (r,1))
    # b is necessarily the rest of the rows
    b = 1-a
    child = a*expander1 + b*expander2
    return child

def mutate(expander):
    r,c = expander.shape
    temp_row = randint(0,r)
    temp = expander.copy()
    col_mix = choice(np.arange(c), c, replace = False)
    temp[temp_row,:] = temp[temp_row,col_mix]
    return temp

def create_initial_population(expander):
    """
    Instead of doing these as arrays
    I'll take advantage of Python's list
    comprehension and give all populations
    as lists.  This will also account
    for the slightly different take on
    the breed and mutate functions.
    """
    pop0 = [mix_columns(expander) for k in range(64)]
    pop0.append(expander)
    for n in range(2):
        pop0.append(anneal(expander)[0])
        pop0.append(expander_transposer(expander, 500))
        print("added member {0}".format(17+n))
    return pop0

def create_new_population(old_population, size_of_new_population = 40):
    for n in range(size_of_new_population):
        pair = randint(0,len(old_population),2)
        child1 = breed(old_population[pair[0]],old_population[pair[1]])
        child2 = breed(old_population[pair[0]],old_population[pair[1]])
        mutation1 = mutate(old_population[pair[0]])
        mutation2 = mutate(old_population[pair[1]])
        mutation3 = mix_columns(child1)
        mutation4 = mix_columns(child2, child2.shape[0]//2)
        old_population.append(child1)
        old_population.append(child2)       
        old_population.append(mutation1)
        old_population.append(mutation2)
        old_population.append(mutation3)
        old_population.append(mutation4)
    fits = [fitness2(old_population[k]) for k in range(len(old_population))]
    best_n = np.argsort(fits)[:size_of_new_population]
    new_population = [old_population[best_n[n]] for n in range(size_of_new_population)]
    return new_population, np.min(fits)     

def evolve_to_solution(expander, num_generations, size_generation = 40):
    pop = create_initial_population(expander)
    best_fits = [fitness2(expander)]
    for gen in range(num_generations):
        pop, new_fit = create_new_population(pop, size_generation)
        best_fits.append(new_fit)
        if gen > 100:
            if best_fits[-99] == best_fits[-1]:
               print("Fell into a local minimum after {0} generations".format(gen))
               break
        if gen%50 == 0:
            print("Completed {0} generations".format(gen + 1))
            print("Current best fit is {0}".format(new_fit))
    return pop[0], pop, best_fits[2:]    
    
