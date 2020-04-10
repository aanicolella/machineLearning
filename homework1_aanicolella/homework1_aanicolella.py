import numpy as np
from numpy import linalg as la

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return np.dot(A,B) - C

def problem3 (A, B, C):
    return A * B + np.transpose(C)

def problem4 (x, y):
    return np.inner(x,y)

def problem5 (A):
    return np.zeros(A.size).reshape(A.shape)

def problem6 (A):
    return np.ones(A.shape[0]).reshape(A.shape[0],1)

def problem7 (A, alpha):
    return A + alpha * np.eye(A.shape[0])

def problem8 (A, i, j):
    return A[i][j]

def problem9 (A, i):
    return np.sum(A[i])

def problem10 (A, c, d):
    return np.mean(np.nonzero(A>=c, A, A<=d))

def problem11 (A, k):
    eig = la.eig(A)[1]
    numCol = A.shape - k
    return eig[:,numCol:]

def problem12 (A, x):
    return la.solve(A,x)

def problem13 (A, x):
    solveTrans = la.solve(np.transpose(A), np.transpose(x))
    return np.transpose(solveTrans)
