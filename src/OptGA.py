# Optimization Method: Genetic Algorithm
# Origin:   Mathematic Modeling
# Author:   Pete
# Date:     2019-11-18

# Description:
#   input:
#       f_sym: function (symbol)
#       x_sym: variable (symbol)
#       x_min: minimum value of each variable (list)
#       x_max: maximum value of each variable (list)
#   output:
#       f_val: optimum point's value
#       x_cur: optimum point's location
#       x_rec: search path record

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sympy import Symbol as syms
from sympy import diff
from sympy import exp
from operator import itemgetter
from random import randint
from random import uniform

def GA_func(f_sym,x_sym,x_min,x_max):
    x_val = np.array([np.linspace(x_min[i],x_max[i],10) for i in range(len(x_sym))]).T

    gen = 0
    gen_max = 2000
    while True:
        pop_num,var_num = x_val.shape

        f_val = []
        for p in range(pop_num):
            f_tmp = f_sym
            for s,v in list(zip(x_sym,x_val[p])):
                f_tmp = f_tmp.subs(s,v)
            f_val.append(f_tmp)
        f_val = np.array(f_val).reshape(pop_num)

        f_fit = []
        for p in range(pop_num):
            f_fit.append(custumized_fitness_func(f_val[p]))
        f_fit = np.array(f_fit).reshape(pop_num)

        if gen > gen_max:
            break
        else:
            gen = gen + 1

        x_val = inherit_func(f_fit,x_val)
        x_val = mutate_func(f_fit,x_val,x_min,x_max)

    return [f_val[0],x_val[0]]

def custumized_fitness_func(f_val):
    return 1 / f_val**2

def minimize_fitness_func(f_val,f_max,Iter):
    return f_max - f_val + 0.99**Iter

def maximize_fitness_func(f_val,f_min,Iter):
    return f_val - f_min + 0.99**Iter

def inherit_func(f_fit,x_val):
    d = np.array(sorted(list(zip(f_fit,x_val)),key=lambda x:x[0],reverse=True))
    sorted_fit = np.array(list(d[:,0]))
    sorted_val = np.array(list([list(x) for x in d[:,1]]))

    save_num = 3
    for i in range(save_num,len(d)):
        if uniform(0,1) > 0.5:
            continue

        r_row = [randint(save_num,x_val.shape[0]-1), randint(save_num,x_val.shape[0]-1)]
        r_col = [randint(0,       x_val.shape[1]  ), randint(0,       x_val.shape[1]  )]
        r_row.sort()
        r_col.sort()
        
        temp                                   = sorted_val[r_row[0],r_col[0]:r_col[1]].copy()
        sorted_val[r_row[0],r_col[0]:r_col[1]] = sorted_val[r_row[1],r_col[0]:r_col[1]].copy()
        sorted_val[r_row[1],r_col[0]:r_col[1]] =                                   temp.copy()
    
    return sorted_val

def mutate_func(f_fit,x_val,x_min,x_max):
    d = np.array(sorted(list(zip(f_fit,x_val)),key=lambda x:x[0],reverse=True))
    sorted_fit = np.array(list(d[:,0]))
    sorted_val = np.array(list([list(x) for x in d[:,1]]))

    save_num = 3
    for i in range(save_num,len(d)):
        if uniform(0,1) > 0.3:
            continue

        r_row = randint(save_num, x_val.shape[0]-1)
        r_col = randint(0,        x_val.shape[1]-1)

        sorted_val[r_row,r_col] = uniform(x_min[r_col],x_max[r_col])

    return sorted_val

if __name__ == "__main__":
    x1 = syms('x1')
    x2 = syms('x2')
    f = x1*x1 + x2*x2
    x_sym = [x1,x2]
    x_min = [-5,-5]
    x_max = [ 5, 5]
    f_val,x_val = GA_func(f,x_sym,x_min,x_max)

    print("optimum value: ",f_val)
    print("optimum point: ",x_val)
