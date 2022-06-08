import numpy as np
from scipy.spatial import distance

def KL(a,b):
    return np.sum(np.multiply(a,np.log(np.divide(a,b)))) 

def Euclidean(a, b):
    return np.linalg.norm(a-b)

def jsd(a, b):
    return distance.jensenshannon(a, b)

def summary(s1, s2):
    M = (s1 + s2) / 2
    KL1 = KL(s1, M)
    KL2 = KL(s2, M)
    return (KL1 + KL2) / 2