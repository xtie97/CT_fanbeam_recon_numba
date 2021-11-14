# Fan_beam_Recon_Curved/Linear_Detector
import numpy as np
import cupy as cp
from config import config
import matplotlib.pylab as plt
import math 
from numba import cuda 
import operator 
import time 
from utils import read_binary_dataset, numba_cuda_conv 
from utils import conv_kernel_ramp_curved

@cuda.jit
def RayFBP_curved(prj, img, angle, angle_index, SID, SDD, Xmin, pX, ndet, ddet, dtheta): 
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
    nX = img.shape[1]
    rfov = nX*pX/2
    
    if (pos_z < img.shape[0]):
        x_pos = Xmin + pX * pos_y
        y_pos = Xmin + pX * pos_x
        if (pow(x_pos, 2) + pow(y_pos, 2) < pow(rfov, 2)): 
            sx_pos = SID * math.cos(angle)
            sy_pos = SID * math.sin(angle)
            L2 = pow((x_pos - sx_pos), 2) + pow((y_pos - sy_pos), 2)
            weights = SID / L2*dtheta
        
            gamma0 = - math.asin( ( -x_pos * math.sin(angle) + y_pos * math.cos(angle)) / math.sqrt(L2) )
            det_index = ( gamma0 + ndet*ddet/2 - ddet/2 ) / ddet
            if (det_index > 0) and (det_index < ndet-1):
                det_index_low = math.floor(det_index)
                w_high = det_index - det_index_low 
                w_low = 1 - w_high
                img[pos_z, pos_x, pos_y] += weights * (prj[pos_z, angle_index, int(det_index_low)] * w_low + prj[pos_z, angle_index, int(det_index_low)+1] * w_high)
        
def fbp_fan_curved_recon(device_idx=0):
    filename = config.ProjectionFilename
    prj = read_binary_dataset(filename, [config.ProjectionsPerRotation, config.DetectorColumnNumber])
    
    # Scanner geometry
    nslice = config.NZ
    SDD = config.SourceDetectorDistance 
    SID = config.SourceAxisDistance
    
    # Image geometry: 
    rfov = config.FOVRadius
    nX = config.NX 
    nY = config.NY 
    pX = rfov*2/nX 
    pY = rfov*2/nY 
    Xmin = -nX/2*pX + pX/2
    Ymin = -nY/2*pY + pY/2
    
    # Detector geometry: 
    nfan = config.FanAngle / 180 * math.pi 
    ndet = config.DetectorColumnNumber 
    ddet = nfan/ndet
    gamma = np.arange(ndet, dtype='float32') * ddet - nfan/2 + ddet/2
    
    prj *= np.cos(gamma) # pre-weighting 
    recon_kernel = conv_kernel_ramp_curved((ndet-1)*2+1, ndet-1, ddet)
    
    # Views: 
    nview = config.ProjectionsPerRotation 
    total_angle = config.FullAngle/180*math.pi  # full scan / short scan / super short scan
    dtheta = total_angle/nview 
    theta = config.StartAngle/180*math.pi + np.arange(nview, dtype='float32') * dtheta # [0, total angle)
    recon_kernel *= ddet 
    
    # Proceed the convolution / filtering
    prj_pad1 = np.zeros((nslice, nview, ndet-1), dtype='float32')
    prj_pad2 = np.zeros((nslice, nview, ndet-1), dtype='float32')
    prj_pad = np.concatenate((prj_pad1, prj, prj_pad2), axis=2)
    cuda.select_device(device_idx)
    stream = cuda.stream()
    prj_pad_device = cuda.to_device(prj_pad, stream=stream)
    kernel_device = cuda.to_device(recon_kernel, stream=stream)
    prj_filter = cuda.device_array_like(prj, stream=stream)
    
    threadsperblock = (32, 32, 1)
    blockspergrid_x = math.ceil(prj.shape[1] / threadsperblock[0]) # nview / 32
    blockspergrid_y = math.ceil(prj.shape[2] / threadsperblock[1]) # ndet / 32
    blockspergrid_z = math.ceil(prj.shape[0] / threadsperblock[2]) # nrow / 1
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    numba_cuda_conv[blockspergrid, threadsperblock, stream](prj_pad_device, kernel_device, prj_filter)
    
    cuda.synchronize()
    img = np.zeros((nslice, nX, nY), dtype='float32')
    img_cuda = cuda.to_device(img, stream=stream)
    
    threadsperblock = (32, 32, 1)
    blockspergrid_x = math.ceil(img.shape[1] / threadsperblock[0]) # img.nx / 32
    blockspergrid_y = math.ceil(img.shape[2] / threadsperblock[1]) # img.ny / 32
    blockspergrid_z = math.ceil(img.shape[0] / threadsperblock[2]) # nrow / 1
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    for angleidx in range(nview):
        angle = theta[angleidx]
        RayFBP_curved[blockspergrid, threadsperblock](prj_filter, img_cuda, angle, angleidx, SID, SDD, Xmin, pX, ndet, ddet, dtheta)
        cuda.synchronize()
         
    img_cuda.copy_to_host(img) 
    img /= 2 # full scan (redundant factor)
    
    f = open(config.ReconstructionFilename, "wb")
    f.write(bytearray(img))
    f.close()
    cuda.close()

if __name__ == "__main__":
    t_start = time.time()
    if config.DetectorType == 1: # curved 
        fbp_fan_curved_recon()
    elif config.DetectorType == 0: # linear 
        pass
    else:
        raise Exception("This type of detector is not included")
    t_end = time.time()
    print('Running time: {}s'.format(t_end-t_start) )


