import numpy as np
import cupy as cp
from config import config
import matplotlib.pylab as plt
import math 
from numba import cuda 
import operator 
import time 

def read_binary_dataset(file_name, dim):   # please pay attentions to the order of data!\n"
    f = open(file_name, 'rb')
    data = np.fromfile(f, dtype=np.float32, count=-1)
    f.close()
    data = np.reshape(data, (-1, dim[0], dim[1]), order='C') # -1: slice 
    return data
    
def conv_kernel_ramp_curved(ksize, det_center, ddet):
    H_ramp = np.zeros((1, ksize), dtype='float32')
    H_ramp[0, det_center] = 1/(2*ddet)**2 # 0 
    nonzero_index = np.arange((det_center-1)%2, ksize, 2) # odd
    H_ramp[0, nonzero_index] = -1 / ( math.pi * np.sin((nonzero_index-det_center)*ddet) )**2
    return H_ramp.squeeze()
    
def conv_kernel_ramp_linear(ksize, det_center, ddet):
    H_ramp = np.zeros((1, ksize), dtype='float32')
    H_ramp[0, det_center] =  1/(2*ddet)**2  # 0 
    nonzero_index = np.arange((det_center-1)%2, ksize, 2) # odd
    H_ramp[0, nonzero_index] = -1 / (ddet)**2 / (math.pi*(nonzero_index - det_center))**2
    return H_ramp.squeeze()
      
@cuda.jit
def numba_cuda_conv(x, k, y): # 1D convolution 
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    # Block id in a 2D grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z
    # Block width, i.e. number of threads per block
    bwx = cuda.blockDim.x
    bwy = cuda.blockDim.y
    bwz = cuda.blockDim.z
    # Compute flattened index inside the array
    pos_x = tx + bx * bwx  
    pos_y = ty + by * bwy
    pos_z = tz + bz * bwz
    if (pos_z < y.shape[0]) and (pos_x < y.shape[1]) and (pos_y < y.shape[2]):
        for j in range(k.size):
            y[pos_z, pos_x, pos_y] += x[pos_z, pos_x, pos_y+j] * k[j]

