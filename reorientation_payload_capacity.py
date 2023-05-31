from tensegrity.tensegrity_plotter import TensegrityPlotter
from tensegrity.tensegrity import Tensegrity
from tensegrity.tensegrity_rotation import TensegrityRotation
from py3dmath.py3dmath import Vec3,Rotation
from reorient.payload_capacity_analyzer import ReorientationPayloadAnalyzer
import matplotlib.pyplot as plt


"""
Parameter for tensegrity aerial vehicle.
The default value here corresponds to the tensegrity aerial vehicle presented in the paper.
Please adjust the following so it matches your vehicle. 
"""
rodLength = 0.20 #Length of tensegrity rod
propX = 0.05 #Offset x-distance for propeller 
propY = 0.034 #Offset y-distance for propeller
torqueFromForceFwd = 0.009975  # [m]
torqueFromForceRev = 0.0163  # [m] 
maxThrust = 2.8  # [N] 
minThrust = -1.5  # [N]
propRadius =  0.032 #[m] 2.5Inch prop
mass = 0.30  # [kg]
frictionCoeff = 0.2

inertia_xx = 7.82e-4 #[kg*m^2]
inertia_yy = 12.59e-4 #[kg*m^2]
inertia_zz = 12.90e-4 #[kg*m^2]
COMOffset = Vec3(0,0,0.01)

motorParam = [minThrust, maxThrust, torqueFromForceFwd, torqueFromForceRev]
inertialParam = [mass, inertia_xx, inertia_yy, inertia_zz]
myTensegrity = Tensegrity(rodLength,propX,propY,propRadius)
myTensegrityRot = TensegrityRotation(myTensegrity,motorParam,inertialParam, COMOffset)
myTensegrityPlotter = TensegrityPlotter(myTensegrity,myTensegrityRot)
myReorientationCapacityAnalyzer = ReorientationPayloadAnalyzer(myTensegrity, motorParam, inertialParam, frictionCoeff)
feasibilityMap = myReorientationCapacityAnalyzer.create_payload_capacity_map()