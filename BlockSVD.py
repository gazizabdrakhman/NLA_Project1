#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"execution time: {end_time - start_time} seconds")
        return result
    return wrapper


# # Implementation from "Vinita Vasudevan, M. Ramakrishna, A hierarchical singular value decomposition algorithm for low rank matrices" https://arxiv.org/abs/1710.02812

# In[98]:


@timeit
def blockSVD(A, d, c, gamma = 0.0000001):
    V_hat, S_hat = DoSVDofBlocks(A, d, c, gamma)
    U_hat, S_hat, V_hat = Iter_improve(A, V_hat, S_hat, gamma)
    print("rank of matrix approximation is: ", S_hat.shape[0])
    return U_hat, S_hat, V_hat


# # Algorithm 1: construct $\hat{V}, \hat{\Sigma}$

# In[99]:


def DoSVDofBlocks(A, d, c, gamma):
    Nd = int((A.shape[0]/d) + 0.5)
    list_V = [] 
    list_S = []
    list_X = row_partition(A, d)
    for j in range(Nd):
        U_j, S_j = DoSVDofColSlices(list_X[j], c, gamma)
        _, S_j, VT_j = tr_SVD(U_j.T@list_X[j], gamma)
        list_V.append(VT_j.T), list_S.append(S_j) #calculating V, not V.T!
    return DoMergeOfSlices(list_V, list_S, gamma)


def DoSVDofColSlices(A, c, gamma):
    list_X = col_partition(A, c)
    list_U = []
    list_S = []
    Nc = int((A.shape[1]/c) + 0.5) 
    for j in range(Nc):
        U_j, S_j, _ = tr_SVD(list_X[j], gamma)
        list_U.append(U_j)
        list_S.append(S_j)
    U_hat, S_hat = DoMergeOfSlices(list_U, list_S, gamma)
    return U_hat, S_hat


def DoMergeOfSlices(list_U, list_S, gamma):
    levels = int((np.log2(len(list_U)))+0.5)
    for j in range(levels):
        N1 = len(list_U)
        list_Ut = list_U 
        list_St = list_S
        list_U = []
        list_S = []
        for i in range(1,N1,2):
            U_j, S_j, _ = BlockMerge(list_Ut[i-1], list_St[i-1], list_Ut[i], list_St[i], gamma)
            list_U.append(U_j), list_S.append(S_j)
        if N1 % 2 == 1:
            list_U.extend(list_Ut[i+1:]), list_S.extend(list_St[i+1:])
    return list_U[0], list_S[0]


def BlockMerge(U1, Sigma1, U2, Sigma2, gamma):
    
    n_rows = Sigma2.shape[0]
    n_cols = Sigma1.shape[0]
    
    Q = U2 - U1@U1.T@U2
    U0, R = np.linalg.qr(Q, 'reduced')
    E = np.block(
        [[np.diag(Sigma1), U1.T@U2@np.diag(Sigma2)], \
        [np.zeros((n_rows, n_cols)), R@np.diag(Sigma2)]]
                )
    UE, SE, VET = tr_SVD(E, gamma) 
    return np.block([[U1, U0]])@UE, SE, VET


# # Algorithm 2: construct more precise $\hat{U}, \hat{\Sigma}, \hat{V}$ using $\hat{V}, \hat{\Sigma}$ from Algorithm 1.

# In[100]:


def Iter_improve(A, V_hat, S_hat, gamma):
    while True:
        U_tilde_i, S_tilde_i, VT_tilde_i = tr_SVD(A@V_hat, gamma)
        U_tilde, S_tilde, VT_tilde = tr_SVD((U_tilde_i.T)@A, gamma)
        Error = np.linalg.norm(np.diag(S_hat - S_tilde), \
                ord = 2)/np.linalg.norm(np.diag(S_hat), ord = 2)
        S_hat = S_tilde
        if Error < 1e-2:
            break
            
    return U_tilde_i@U_tilde, S_hat, VT_tilde.T


# ### Supporting functions like truncated SVD using a fraction of maximum singular value, row and column partitions of an mxn matrix.

# In[101]:


def tr_SVD(A, gamma):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    num_kept = retain(S, gamma)
    U_tr = U[:, :num_kept]
    S_tr = S[:num_kept]
    VT_tr = VT[:num_kept, :]
    return U_tr, S_tr, VT_tr


def retain(Sigma, gamma): 
    sigma_max = Sigma[0]
    non_zero_Sigma = Sigma[~np.isclose(Sigma, 0)].copy()
    return np.sum(non_zero_Sigma >= sigma_max*gamma)


def col_partition(Matrix, c):
    list_X = []
    n = Matrix.shape[1] 
    num_partitions = int((n/c)+0.5)
    
    for i in range(num_partitions):
        start_col = i * c
        end_col = (i + 1) * c
        submatrix = Matrix[:, start_col:end_col]
        list_X.append(submatrix)
    return list_X

def row_partition(Matrix, d):
    list_X = []
    m = Matrix.shape[0] 
    num_partitions = int((m/d)+0.5)

    for i in range(num_partitions):
        start_row = i * d
        end_row = (i + 1) * d
        submatrix = Matrix[start_row:end_row, :]
        list_X.append(submatrix)
    return list_X


# ### Functions for matrix construction and quality evaluation

# In[102]:


def construct_matrix(m=1000,n=500, \
        singular_values=[1,0.8,0.7,0.5,0.1,0.05,\
        0.01,0.001, 0.0000001, 0.0000000001], seed=42):
    
    np.random.seed(seed)
    S = np.diag(singular_values)  
    rank = S.shape[0]
    U, _ = np.linalg.qr(np.random.rand(m, rank))
    V, _ = np.linalg.qr(np.random.rand(n, rank))
    A = U@S@V.T
    print("matrix has memory of ", A.nbytes/(1024 * 1024), "megabytes")
    return A

def quality(A, U_aprx, S_aprx, V_aprx):
    print('relative error is: ', np.linalg.norm(U_aprx@np.diag(S_aprx)@V_aprx.T - A)/np.linalg.norm(A))
    print("np.allclose is: ", np.allclose(U_aprx@np.diag(S_aprx)@V_aprx.T, A))


# ## Example:

# In[103]:


if __name__ == "__main__":
    A = construct_matrix(m=100, n=500, singular_values=\
    [1, 0.8, 0.7, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0000001, 0.0000000001])
    U_aprx, S_aprx, V_aprx = blockSVD(A, 10, 10, gamma=0.1)
    quality(A, U_aprx, S_aprx, V_aprx)

