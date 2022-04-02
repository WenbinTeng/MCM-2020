# Optimization Method: Simulated Annealing Algorithm
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

def SAA_func(f_sym,x_sym,x_min,x_max):
    x_val = [uniform(xmin,xmax) for xmin,xmax in list(zip(x_min,x_max))]
    x_opt = x_val.copy()
    x_old = x_val.copy()
    x_new = x_val.copy()

    f_val = f_sym
    for s,v in list(zip(x_sym,x_val)):
        f_val = f_val.subs(s,v)
    f_val = float(f_val)
    f_opt = f_val
    f_old = f_val
    f_new = f_val

    de_rate = 0.99
    rand_min = [-1,-1]
    rand_max = [ 1, 1]
    t_cur = 100
    t_end = 1
    while t_cur > t_end:
        x_old = x_new.copy()
        x_new = [x_new[x] + uniform(rand_min[x],rand_max[x]) for x in range(len(x_sym))]

        f_old = f_new
        f_new = f_sym
        for s,v in list(zip(x_sym,x_new)):
            f_new = f_new.subs(s,v)
        f_new = float(f_new)
        
        if f_new < f_old:
            x_opt = x_new.copy()
            f_opt = f_new
            continue
        elif uniform(0,1) < np.exp(-(f_new-f_old)/t_cur):
            continue
        else:
            x_new = x_old.copy()
            f_new = f_old

        t_cur = t_cur * de_rate
        
    return [f_opt,x_opt]

if __name__ == "__main__":
    x1 = syms('x1')
    x2 = syms('x2')
    f = x1*x1 + x2*x2
    x_sym = [x1,x2]
    x_min = [-5,-5]
    x_max = [ 5, 5]
    f_val,x_val = SAA_func(f,x_sym,x_min,x_max)

    print("optimum value: ",f_val)
    print("potimum point: ",x_val)