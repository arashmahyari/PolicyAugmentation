# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:18:38 2020

@author: arash
"""

import numpy as np
from scipy.stats import ortho_group




def IMCMan(Ut1,Vt1,A,Nq,X,Y):
    # Algorithm 1, paper: Robust Inductive Matrix Completion Strategy
    
    lambda1=0.5
    lambda2=0.5
    
    A=np.divide(np.array(A),Nq)
    X=np.array(X)
    Y=np.array(Y)
    
    [M,m]=np.shape(X)
    [N,n]=np.shape(Y)
    
    
    Wo=np.random.uniform(size=(m,1))
    Ho=np.random.uniform(size=(n,1))
    
    Con=True
    while Con:
        aa=np.linalg.pinv(np.matmul(Vt1,np.transpose(Vt1)))
        #print(np.shape(Ho))
        a=np.matmul(aa,Ho)
        
        
        K1=np.matmul(np.transpose(Y),np.matmul(np.transpose(A),np.matmul(X,Wo))) / (np.matmul(np.matmul(np.matmul(np.transpose(Y),Y),np.matmul(Ho,np.transpose(Wo))),np.matmul(np.transpose(X),np.matmul(X,Wo)))+ lambda2*Ho +a)
        H=np.multiply(Ho,K1)
        
        a=np.matmul(np.linalg.pinv(np.matmul(Ut1,np.transpose(Ut1))),Wo)
        K2=np.matmul(np.transpose(X),np.matmul(A,np.matmul(Y,H))) / (np.matmul(np.matmul(np.matmul(np.transpose(X),X),np.matmul(Wo,np.transpose(H))),np.matmul(np.transpose(Y),np.matmul(Y,H)))+ lambda1*Wo+a)
        W=np.multiply(Wo,K2) 
        
        Con=(np.linalg.norm(Ho-H) < 0.01) and (np.linalg.norm(Wo-W) < 0.01)
        Ho=H
        Wo=W
    
    M=np.matmul(np.matmul(X,W),np.transpose(np.matmul(Y,H)))
    
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if np.sum(A[i,j])!=0:
                M[i,j]=A[i,j]
    
    
    return M, W, H
       
    
    


def Xr(A,U,V,X,Y):
    Mas=A!=0
    #print(np.shape(X))
    #print(np.shape(Y))
    #print(np.shape(U))
    #print(np.shape(V))
    
    b=np.matmul(X,U)
    c=np.transpose(np.matmul(Y,V))
    a=np.matmul(b,c)
    return np.multiply(A-a,Mas)







    
    
