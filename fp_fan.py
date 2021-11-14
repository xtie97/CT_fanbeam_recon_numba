# Fan_beam_Recon_Curved/Linear_Detector
import numpy as np
import cupy as cp
from config import config
import matplotlib.pylab as plt
import math 
from numba import cuda 
import operator 
import time 
from utils import read_binary_dataset 

@cuda.jit
def RayFP_curved(img, prj, angle_start, SID, pxsize, ndet, ddet, dtheta, stepsize, nlength): 
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z 
    
    bwx = cuda.blockDim.x
    bwy = cuda.blockDim.y
    bwz = cuda.blockDim.z
    
    theta_idx = tx + bx * bwx # x: view
    gamma_idx = ty + by * bwy # y: ray 
    pos_z = tz + bz * bwz
    
    gamma_start = - ndet*ddet/2 + ddet/2
    theta_start = angle_start/180*math.pi 
    
    nX = img.shape[1]
    nY = img.shape[2]
    rfov = nX/2 * pxsize
    fov_extend = rfov + stepsize * 10
    SROID = SID - fov_extend # source to the edge of ROI distance
    
    if (pos_z < prj.shape[0]) and (theta_idx < prj.shape[1]) and (gamma_idx < prj.shape[2]):
        gamma = gamma_start + gamma_idx*ddet
        theta = theta_start + theta_idx*dtheta
        sx = SID*math.cos(theta)
        sy = SID*math.sin(theta)
        for alpha in range(nlength):
            px = sx - (SROID + alpha*stepsize) * math.cos(gamma+theta)
            py = sy - (SROID + alpha*stepsize) * math.sin(gamma+theta)
            pxind = py/pxsize + nY/2 - 0.5
            pyind = px/pxsize + nX/2 - 0.5 
            
            px1 = math.floor(pxind)
            py1 = math.floor(pyind)
            if (pxind - px1) < 0.5:
                px2 = px1 - 1
            else:
                px2 = px1 + 1
                
            if (pyind - py1) < 0.5:
                py2 = py1 - 1
            else:
                py2 = py1 + 1
                
            if (min(px1, px2)>-1) and (max(px1, px2)<nX) and (min(py1, py2)>-1) and (max(py1, py2)<nY):
                w1 = (py2 - pyind)/(py2-py1) * stepsize 
                w2 = (pyind - py1)/(py2-py1) * stepsize 
                r1 = ( (px2 - pxind)/(px2-px1) * img[pos_z, int(px1), int(py1)] + (pxind - px1)/(px2-px1) * img[pos_z, int(px2), int(py1)] ) * w1
                r2 = ( (px2 - pxind)/(px2-px1) * img[pos_z, int(px1), int(py2)] + (pxind - px1)/(px2-px1) * img[pos_z, int(px2), int(py2)] ) * w2
                prj[pos_z, theta_idx, gamma_idx] += r1 + r2
                
def fp_fan_curved(device_idx=0):
    filename = config.ProjectionFilename
    img = read_binary_dataset(filename, [config.NX, config.NY])
    # Scanner geometry
    nslice = config.NZ
    SDD = config.SourceDetectorDistance 
    SID = config.SourceAxisDistance
    
    # Image geometry: 
    rfov = config.FOVRadius
    nX = config.NX 
    nY = config.NY
    pX = rfov*2/nX 
    Xmin = -nX/2*pX + pX/2
    
    # Detector geometry: 
    nfan = config.FanAngle / 180 * math.pi 
    ndet = config.DetectorColumnNumber 
    ddet = nfan/ndet
    gamma = np.arange(ndet, dtype='float32') * ddet - nfan/2 + ddet/2
    
    # Views: 
    nview = config.ProjectionsPerRotation 
    total_angle = config.FullAngle/180*math.pi  # full scan / short scan / super short scan
    dtheta = total_angle/nview 
    
    # Ray tracing interpolation:
    stepsize = 0.4 # mm
    nlength = int( ((rfov+SID)/math.cos(nfan/2) - (SID-rfov-stepsize*10)) /stepsize )
    
    cuda.select_device(device_idx)
    stream = cuda.stream()
    prj = np.zeros((nslice, nview, ndet), dtype='float32')
    prj_device = cuda.to_device(prj, stream=stream)
    
    img_device = cuda.to_device(img, stream=stream)
    
    threadsperblock = (32, 32, 1)
    blockspergrid_x = math.ceil(prj.shape[1] / threadsperblock[0]) # nview / 32
    blockspergrid_y = math.ceil(prj.shape[2] / threadsperblock[1]) # ndet / 32
    blockspergrid_z = math.ceil(prj.shape[0] / threadsperblock[2]) # nslice / 1 
    
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    RayFP_curved[blockspergrid, threadsperblock](img_device, prj_device, config.StartAngle, SID, pX, ndet, ddet, dtheta, stepsize, nlength)
    cuda.synchronize()   
    
    prj_device.copy_to_host(prj)  
    f = open(config.ReconstructionFilename, "wb")
    f.write(bytearray(prj))
    f.close()
    cuda.close()
    
    
if __name__ == "__main__":
    t_start = time.time()
    if config.DetectorType == 1: # curved 
        fp_fan_curved()
    elif config.DetectorType == 0: # linear 
        pass
    else:
        raise Exception("This type of detector is not included")
    t_end = time.time()
    print('Running time: {}s'.format(t_end-t_start) )


