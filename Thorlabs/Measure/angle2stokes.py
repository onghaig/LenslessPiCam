import numpy as np
import pandas as pd


def stokes_from_alphabeta(alpha,beta):
    # fast axis of HWP and QWP should be vertical
    # alpha: an array of HWP angels, in rad
    # beta: an array of QEP angels, in rad
    # return array(n,3), each row represent [S1/S0,S2/S0,S3/S0]
    result = np.zeros((alpha.shape[0],3),dtype=np.float32)
    result[:,0] = 0.5*(np.cos(4*alpha-4*beta)+np.cos(4*alpha))
    result[:,1] = 0.5*(np.sin(4*alpha)-np.sin(4*alpha-4*beta))
    result[:,2] = -np.sin(4*alpha-2*beta)
    return result