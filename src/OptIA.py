# Optimization Method: Immune Algorithm
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

def IA_func(f_sym,x_sym,x_min,x_max):
    x_val = np.array([np.linspace(x_min[i],x_max[i],10) for i in range(len(x_sym))]).T

    gen = 0
    gen_max = 100
    while True:
        pop_num,var_num = x_val.shape

        x_inf = np.zeros(pop_num)
        for n in range(pop_num):
            for i in range(pop_num):
                num = 0
                for j in range(var_num):
                    if abs(x_val[n,j]-x_val[i,j]):
                        num = num + 1
                if num == 0 or num == var_num:
                    continue
                else:
                    x_inf[n] = x_inf[n] + num / var_num * np.log2(var_num/num)

        x_rep = np.zeros((pop_num,pop_num))
        for i in range(pop_num):
            for j in range(pop_num):
                if i > j:
                    x_rep[i,j] = x_rep[j,i]
                else:
                    x_rep[i,j] = 1 / (1 + 0.5 * (x_inf[i] + x_inf[j]))

        f_val = []
        for p in range(pop_num):
            f_tmp = f_sym
            for s,v in list(zip(x_sym,x_val[p])):
                f_tmp = f_tmp.subs(s,v)
            f_val.append(f_tmp) 
        f_val = np.array(f_val).reshape(pop_num)

        f_att = []
        for p in range(pop_num):
            f_att.append(custumized_attraction_func(f_val[p]))
        f_att = np.array(f_att).reshape(pop_num)

        if gen > gen_max:
            break
        else:
            gen = gen + 1

        x_val = differentiation_func(f_att,x_val,x_rep,x_min,x_max)

    return [f_val[0],x_val[0]]

def custumized_attraction_func(f_val):
    return 1 / f_val**2

def minimize_attraction_func(f_val,f_max,Iter):
    return f_max - f_val + 0.99**Iter

def maximize_attraction_func(f_val,f_min,Iter):
    return f_val - f_min + 0.99**Iter

def differentiation_func(f_att,x_val,x_rep,x_min,x_max):
    pop_num,var_num = x_val.shape

    att_estimate = f_att / np.max(f_att)
    att_thr = 0.5
    att_val = 0
    for i in range(pop_num):
        if att_estimate[i] > att_thr:
            att_val = att_val - att_estimate[i]
        att_val = att_val + 1

    rep_estimate = x_rep / np.amax(x_rep)
    rep_thr = 0.5
    rep_val = [0] * pop_num
    for i in range(pop_num):
        for j in range(pop_num):
            if rep_estimate[i,j] > rep_thr:
                rep_val[i] = rep_val[i] + 1 / pop_num

    for i in range(pop_num):
        if uniform(0,1) > (1/rep_val[i]) * (f_att[i]/sum(f_att)) * att_val:
            for j in range(var_num):
                x_val[i,j] = uniform(x_min[j],x_max[j])

    return x_val

if __name__ == "__main__":
    x1 = syms('x1')
    x2 = syms('x2')
    f = x1*x1 + x2*x2
    x_sym = [x1,x2]
    x_min = [-5,-5]
    x_max = [ 5, 5]
    f_val,x_val = IA_func(f,x_sym,x_min,x_max)

    print("optimum value: ",f_val)
    print("potimum point: ",x_val)