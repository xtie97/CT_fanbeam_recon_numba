from easydict import EasyDict as edict
import json

config = edict()
#[File]
config.ProjectionFilename='./test_phantom_sparse.prj'
config.ReconstructionFilename='./test_phantom_sparse.fbp'

#[Scan]
config.SourceType=1  # 0:Parallel Beam; 1:Cone Beam
config.DetectorType=1 # 0:Linear; 1:Curved
#config.DetectorOffset=0    # 1.25
config.SourceDetectorDistance=946.73233346  # mm # 1117.6, 946.73233346 
config.SourceAxisDistance=538.52 # mm # 889, 538.52
config.StartAngle=0 # in degree

config.ProjectionsPerRotation=120 # 1081, 984
config.DetectorColumnNumber=888 # 2001, 888
config.DetectorCoverage=200.1 # detector coverage  mm 
config.DetectorRowNumber=1
config.FanAngle=55.0059 # fan angle in degree
config.FullAngle=360 # 360: full scan, 180+fan: short scan 
config.ParkWeighting=0 # 0:false; 1:true

#config.DetectorLeft=-27.50295    #-31.151376717603757 # in millimeter or degree  -27.52206577  -27.4448
#config.DetectorDeltaRow=1.0239 #1.096519    # mm
#config.DetectorDeltaCol=0.061943581081081 # mm or degree

#[Image]
#config.XMin=-200 # mm
#config.YMin=-200 # mm  
config.NX=512 # # pixels in x/y direction
config.NY=512
config.NZ=1
config.FOVRadius=250 # mm # 75, 200
config.DeltaZ=0.625	# mm

#[Reconstruction]
config.Algorithm='FBP'
config.HelicalReconMode=1
config.FilterType=1 # S-L


