import numpy as np


def getHjorthActivity(sig):
    return np.var(sig)

def getHjorthMobility(sig):
    dsig = sig[1:]-sig[:-1] #first order derivative
    return np.sqrt(np.var(dsig)/np.var(sig))

def getHjorthComplexity(sig):
    dsig = sig[1:]-sig[:-1]
    return getHjorthMobility(dsig)/getHjorthMobility(sig)
