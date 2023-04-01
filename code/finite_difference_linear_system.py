#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 10:17:43 2023

@author: julio

Implementation of the Finite Difference Method for Elliptic Partial Differential Equations
using the generalization of the Crout Factorization Algorithm
Specifically this solves the Poisson Equation by setting f<>0

"""

import numpy as np
import math
import pandas as pd

# imports Crout Generalization algorithm implementation
from crout_factorization_generalization import Crout_generalization 
#from scipy.linalg import lu

def f(x,y):
    return x*math.exp(y)
    #return x-y
    #return 0

def g(x,y,a,b,c,d):
    if x==a:
        return 0
    if x==b:
        return 2*math.exp(y)
    if y==c:
        return x#0#2*y+x
    if y==d:
        return math.exp(1)*x#10*x**2#200*x
    
def l(i,j,n,m): # relabeling function
    return (i+1)+(m-1-(j+1))*(n-1)-1

def u(x,y):
    return x*math.exp(y)



def finite_difference_linear_system(a,b,c,d, # definition of the 2D region to find the aproximate solution
                                    n,m,     # integers defining the grid
                                    f,g):    # functions f and g , f is the rhs of Poisson eq. and g is the boundary contidion fucntion

    h=(b-a)/n
    k=(d-c)/m
    x=np.linspace(a,b,n+1)
    y=np.linspace(c,d,m+1)
    W_ij=np.zeros(((n-1)*(m-1),(n-1)*(m-1)))
    w=np.zeros((n-1)*(m-1))
    for i in range(n-1):
        for j in range(m-1):
            W_ij[l(i,j,n,m),l(i,j,n,m)]=2*((h/k)**2 +1) # first term of finite-difference equation
            if i!=0 and i!=n-2: # not boundary conditions
                W_ij[l(i,j,n,m),l(i+1,j,n,m)]=-1
                W_ij[l(i,j,n,m),l(i-1,j,n,m)]=-1
            else:
                if i==0: # boundary condition
                    W_ij[l(i,j,n,m),l(i+1,j,n,m)]=-1
                    w[l(i,j,n,m)]+=g(x[i],y[j+1],a,b,c,d)
                    
                if i==n-2: # boundary condition
                    W_ij[l(i,j,n,m),l(i-1,j,n,m)]=-1
                    w[l(i,j,n,m)]+=g(x[i+2],y[j+1],a,b,c,d)
                    
            
            if j!=0 and j!=m-2: #not boundary condition
                W_ij[l(i,j,n,m),l(i,j+1,n,m)]=-(h/k)**2
                W_ij[l(i,j,n,m),l(i,j-1,n,m)]=-(h/k)**2
            else:
                if j==0: # boundary condition
                    W_ij[l(i,j,n,m),l(i,j+1,n,m)]=-(h/k)**2
                    w[(l(i,j,n,m))]+=((h/k)**2)*g(x[i+1],y[j],a,b,c,d)
                    
                if j==m-2: # boundary condition
                    W_ij[l(i,j,n,m),l(i,j-1,n,m)]=-(h/k)**2
                    w[l(i,j,n,m)]+=((h/k)**2)*g(x[i+1],y[j+2],a,b,c,d)
            w[l(i,j,n,m)]+=-(h**2)*f(x[i+1],y[j+1]) # rhs, Poisson equation term
    
    return W_ij, w ,x,y# Returns coefficient matrix W_ij and constant vector w to form a linear systems that feeds the Crout-algorithm
#a=0; b=0.5; c=0; d=0.5; n=4; m=4
#a=0; b=12; c=0; d=12; n=6; m=4
def error_table(n,m,x,y,w,u):
    csv_list=[]
    for i in range(n-1):
        for j in range(m-1):
            element=[i+1,j+1,x[i+1],y[j+1],w[l(i,j,n,m)],u(x[i+1],y[j+1]),abs(u(x[i+1],y[j+1])-w[l(i,j,n,m)])]
            csv_list.append(element)
    column_scheme=['i',
                   'j',
                  'x_i',
                  'y_j',
                 'w_ij',
            'u(x_i,y_j)',
        '|u(x_i,y_j)-w_ij|']
    csv_file_df=pd.DataFrame(csv_list, columns=column_scheme)  
    csv_file_df.to_csv('error_table.csv', index=False)
    return csv_file_df
        
        
a=0.0; b=2.0; c=0.0; d=1.0; n=6; m=5
A,w,x,y=finite_difference_linear_system(a,b,c,d,n,m,f,g)
# solves the system using numpy
sol=np.linalg.solve(A,w)
# solves the system using Crout Factorization for block tridiagonal matrices
sol2=Crout_generalization(A,w,n-1) 
error_analysis_table=error_table(n,m,x,y,sol2,u)