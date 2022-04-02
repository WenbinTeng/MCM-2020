# Optimization Method: Damped Newton Method
# Origin:   Mathematic Modeling
# Author:   Pete
# Date:     2019-11-18

# Description:
#   input:
#       f_sym: function (symbol)
#       x_sym: variable (symbol)
#       x_ini: initial point (list)
#   output:f_val,x_cur,x_rec
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

inf = float('inf')

def Newton_Dam(f_sym,x_sym,x_ini):

    e = 1e-6
    x_rec = np.array(x_ini,dtype=np.float32)
    x_cur = np.array(x_ini,dtype=np.float32)
    g_val = np.zeros(len(x_ini))
    g_sym = [diff(f_sym, x)                    for x in x_sym]
    h_sym = [diff(d_sym, x) for d_sym in g_sym for x in x_sym]
    h_val = np.eye(len(x_ini))
    
    gen = 0
    gen_max = 100
    while gen < gen_max:
        gen = gen + 1

        x_cur = x_cur - np.dot(g_val,np.linalg.inv(h_val)) * search_1d(f_sym,x_sym,x_cur,g_val,h_val)
        x_rec = np.append(x_rec, x_cur).reshape(-1,2)

        g_val = []
        for g in g_sym:
            g_tmp = g
            for s,v in list(zip(x_sym,x_cur)):
                g_tmp = g_tmp.subs(s,v)
            g_val.append(g_tmp)
        g_val = np.array(g_val,dtype=np.float32)

        h_val = []
        for h in h_sym:
            h_tmp = h
            for s,v in list(zip(x_sym,x_cur)):
                h_tmp = h_tmp.subs(s,v)
            h_val.append(h_tmp)
        h_val = np.array(h_val,dtype=np.float32).reshape(len(x_ini),len(x_ini))
        
        if sum(g_val**2) < e:
            break

    f_val = f_sym
    for s,v in list(zip(x_sym,x_cur)):
        f_val = f_val.subs(s,v)

    return [f_val,x_cur,x_rec]

def search_1d(y_sym,x_sym,x_val,g_val,h_val):
    Lambda = 0
    min_value = inf
    min_index = 0
    search_step = 1
    search_section = [0,inf]
    while search_step >= 1e-3:
        Lambda = search_section[0]

        while True:
            x_tmp = x_val - np.dot(g_val,np.linalg.inv(h_val)) * Lambda

            y_val = y_sym
            for s,v in list(zip(x_sym,x_tmp)):
                y_val = y_val.subs(s,v)
        
            if min_value > y_val:
                min_value = y_val
                min_index = Lambda
            else:
                search_section[0] = min_index - search_step
                search_section[1] = min_index + search_step
                break

            Lambda = Lambda + search_step

        search_step = search_step * 0.1
    
    return Lambda

if __name__ == "__main__":
    x1 = syms('x1')
    x2 = syms('x2')
    f = x1*x2*exp(-x1**2-x2**2)
    x_sym = [x1,x2]
    x_ini = [1,0.5]
    f_val,x_cur,x_rec = Newton_Dam(f,x_sym,x_ini)

    fig = plt.figure()
    ax = Axes3D(fig)

    x1 = np.linspace(-3, 3, num=50)
    x2 = np.linspace(-3, 3, num=50)
    X1,X2 = np.meshgrid(x1,x2)
    F = X1*X2*np.exp(-X1**2-X2**2)
    ax.plot_surface(X1,X2,F,cmap=plt.cm.Spectral,rstride=3, cstride=3,zorder=10)

    x1data = np.array(x_rec[:,0],dtype=np.float32)
    x2data = np.array(x_rec[:,1],dtype=np.float32)
    fdata = x1data*x2data*np.exp(-x1data**2-x2data**2)
    ax.plot3D(x1data,x2data,fdata,color='k',zorder=20)
    ax.scatter(x1data,x2data,fdata,c='r',s=50,zorder=30)

    plt.show()