
"Created by Shay Deutsch; @author: shaydeu@math.math.ucla.edu "

import networkx as nx 
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
import seaborn as sb
import sklearn as sk
from scipy.io import loadmat
import scipy
import math
from scipy.linalg import eigh
from networkx.utils import not_implemented_for
from numpy import linalg as LA
import numpy.matlib
import control
from control import dlyap

###
##Packedges required to install 
##https://python-control.readthedocs.io/en/0.9.1/
##Networkx
# Linear algebra (scipy.linalg)

## Input: Similarity matrix/Graph Network
##Outout: Node descriptors, edge descriptors

class GSE_Granular() :
      
     def __init__( self, num_descriptors, basis_dim, A, failed_edges) :
          
        self.num_descriptors = num_descriptors
          
        self.basis_dim = basis_dim

        self.A = A
        
        self.failed_edges = failed_edges
        
    
     def centrality( self, A) :
        
       
        self.A = A 
        
        [m, n] = A.shape

#2. Constrcut betweeness centrality graph
        G =  nx.from_numpy_matrix(A)
        b = nx.edge_betweenness_centrality(G)
        values_b = b.values()
        keys_b = b.keys()
        A2 = np.zeros((np.shape(A)))                  
        for (k,v) in b.items():
             A2[k]=v
        
        y = A2.transpose()
        z   = 10*np.add(A2, y)
    
        BC = z.sum(axis=0)
        BC = BC/np.mean(BC)
       
        return z, BC
    
     def compute_basis(x):
         from numpy import linalg as LA
         from scipy.linalg import eigh
         
         bd, B =  eigh(x)
         
         return bd, B
    
     def compute_norm_lap(self, A):
         
         from numpy import linalg as LA
         self.A = A 
         G =  nx.from_numpy_matrix(A)
         A = nx.normalized_laplacian_matrix(G)
         L_norm = A.todense()
         
         return L_norm
      
     def transform( self, A, z, L):
        
            
          [m, n] = z.shape
          I = np.identity(m)  
          X = dlyap(z,L,I)
          X = X.transpose()
          self.X = X
          return X
    
     def descriptor(self, X, num_descriptors, bd, B):
         "WKS descriptor"
         w = 7
         numEigenfunctions = bd.shape[0]
         Bt = np.transpose(B)
         temp1 = np.multiply(B, B)
         temp2 = np.matmul(Bt, temp1)
    
    
         absoluteEigenvalues = abs(bd)
         l = len(absoluteEigenvalues)-1
         emax = np.log(absoluteEigenvalues)[l]
         emin = np.log(absoluteEigenvalues)[1]   
         s = w*(emax-emin) / num_descriptors
         emin = emin + 2*s
         emax = emax - 2*s
         es = np.linspace(emin, emax, num_descriptors)
         T1 = np.matlib.repmat(np.log(absoluteEigenvalues), num_descriptors, 1)
         T1 = np.transpose(T1)
         T2 = np.matlib.repmat(es, numEigenfunctions, 1)
         T  = np.exp( -numpy.multiply(T1-T2, T1-T2)/(2*s*s))
         temp3 = np.transpose(temp2)
         wks = np.matmul(temp3,T);
         wks =  np.matmul(B,wks) 
         Des = wks
        
         return Des
     
     def concatenate_edges(self, A, des, num_descriptors):
            
         G =  nx.from_numpy_matrix(A)
         num_of_edges = G.number_of_edges()
         
         edge_des = [0]*2*num_descriptors
         edge_des = np.array(edge_des)
       
         for e in G.edges:
             des_1 = des[e[0]]
             des_2 = des[e[1]]
             temp1  = np.concatenate((des_1, des_2))
             temp2  = np.concatenate((des_2, des_1))
             temp3  = temp1 + temp2
             edge_des =  np.c_[(edge_des), (temp3)]
         
         edge_des = edge_des[:,1:]
         return edge_des  
  
  
def main() :    
      
    # Load data (granular network represented as similarity matrix W1)
    import mat73
    annots = loadmat('CZ2.60-Adjacency.mat')
   
    FE = mat73.loadmat ('edge_idx_fail.CZ2_60.mat')
    failed_edges = FE['edge_idx_fail']
   
    A = annots['W1']
    [m, n] = A.shape
    
    np.fill_diagonal(A, 0)
   
    num_descriptors = 800
    basis_dim = n
      
    model = GSE_Granular(num_descriptors, basis_dim, A, failed_edges)
    
    ## Compute edge betweeness centrality graph z
    
    z, BC = model.centrality(A)
    ## compute the normalized Laplacian of edge centrality graph
    L = model.compute_norm_lap(z)
    
    ## compute Sylvester node embedding 
    X = model.transform(A, z, L)
    bd, B =  eigh(X)
    
    Des = model.descriptor(X, num_descriptors, bd, B)  
    ##Construct edge embeddings from nodes embeddings
    Edge_Des = model.concatenate_edges(A, Des, num_descriptors)
    Edge_Des = Edge_Des[:,1:]
    
    return Edge_Des, Des
  
if __name__ == "__main__" : 
      
    Edge_Des, Des = main()
    
