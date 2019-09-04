import numpy as np
import math

# correlation coefficient between rows of two matrices
# a: n x m
# b: p x m
# corr2(a,c): n x p matrix of correlation coefficients
def corr2(a,b):
    aa = a - np.mean(a,axis=1,keepdims=True)
    aa = aa / np.sqrt(np.sum(aa**2,axis=1,keepdims=True))
    bb = b - np.mean(b,axis=1,keepdims=True)
    bb = bb / np.sqrt(np.sum(bb**2,axis=1,keepdims=True))
    return np.dot(aa,bb.T)

