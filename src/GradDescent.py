# Optimization Method: Grad Descent
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

def Grad_Descent(f_sym,x_sym,x_ini):

    e = 1e-6
    x_rec = np.array(x_ini,dtype=np.float32)
    x_cur = np.array(x_ini,dtype=np.float32)
    d_sym = [diff(f_sym, x) for x in x_sym]
    rate = 1

    gen = 0
    gen_max = 100
    while gen < gen_max:
        gen = gen + 1

        d_val = []
        for d in d_sym:
            d_tmp = d
            for s,v in list(zip(x_sym,x_cur)):
                d_tmp = d_tmp.subs(s,v)
            d_val.append(d_tmp)
        d_val = np.array(d_val,dtype=np.float32)
        
        x_cur = x_cur - rate * d_val
        x_rec = np.append(x_rec, x_cur).reshape(-1,2)

        if rate > 0.01:
            rate = rate * 0.99

        if sum(d_val**2) < e:
            break

    f_val = f_sym
    for s,v in list(zip(x_sym,x_cur)):
        f_val = f_val.subs(s,v)

    return [f_val,x_cur,x_rec]

if __name__ == "__main__":
    x1 = syms('x1')
    x2 = syms('x2')
    f = x1*x2*exp(-x1**2-x2**2)
    x_sym = [x1,x2]
    x_ini = [1,0.5]
    f_val,x_cur,x_rec = Grad_Descent(f,x_sym,x_ini)

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