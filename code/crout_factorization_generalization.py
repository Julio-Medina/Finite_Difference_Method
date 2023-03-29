#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:09:16 2023

@author: julio
"""
import numpy as np
"""
Implementation of the Generalization of the Crout Factorization algorithm for tridiagonal block matrices

Bibliography:
    
    Burden, Numerical Analysis
    Varga, Iterative Matrix Analysis


"""
# inputs, coefficient matrix A, constant vector K, block size n
def Crout_generalization(A, K ,n):
    N=A.shape[1]//n # computes number of blocks
    W=[] # empty list initialization
    B_i=A[:n,:n] # main diagonal blocks
    C_i=A[:n,n:n+n] # upper diagonal blocks
    W.append(np.linalg.inv(B_i)@C_i)
    for i in range(2,N):
        B_i=A[(i-1)*n:i*n,(i-1)*n:i*n] # main diagonal blocks
        A_i=A[(i-1)*n:i*n,(i-2)*n:(i-1)*n] # lower diagonal blocks
        C_i=A[(i-1)*n:i*n,(i)*n:(i+1)*n]# upper diagonal blocks
        W.append(np.linalg.inv(B_i-A_i@W[i-2])@C_i)
    B_i=A[:n,:n]
    K_i=K[:n]
    G_i=np.linalg.inv(B_i)@K_i
    G=[]
    G.append(G_i)
    for i in range(2,N+1):
        K_i=K[(i-1)*n:i*n] # constant vector block
        B_i=A[(i-1)*n:i*n,(i-1)*n:i*n]
        A_i=A[(i-1)*n:i*n,(i-2)*n:(i-1)*n]
        G.append(np.linalg.inv(B_i-A_i@W[i-2])@(K_i-A_i@G[i-2]))
    Z=G[:]
    #Z[N-1]=G[N-1]
    for i in range(N-2,-1,-1): # computes final solutions blocks Z
        Z[i]=G[i]-W[i]@Z[i+1]
    
    sol=np.concatenate(Z) # concatenates block solutions
    return sol
        
