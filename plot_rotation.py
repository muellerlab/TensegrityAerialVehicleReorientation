from tensegrity.tensegrity_plotter import TensegrityPlotter
from tensegrity.tensegrity import Tensegrity
from tensegrity.tensegrity_rotation import TensegrityRotation
from py3dmath.py3dmath import Vec3,Rotation
from reorient.planner import Reorientation
import matplotlib.pyplot as plt

"""
Parameter for tensegrity aerial vehicle
"""
rodLength = 0.20 #Length of tensegrity rod
propX = 0.05 #Offset x-distance for propeller 
propY = 0.034 #Offset y-distance for propeller
torqueFromForceFwd = 0.009975  # [m]
torqueFromForceRev = 0.0163  # [m] 
maxThrust = 2.5  # [N] 
minThrust = -1.5  # [N]
propRadius =  0.032 #[m] 2.5Inch prop
coeffFriction = 0.2 # [Mu]
mass = 0.282  # [kg]
inertia_xx = 92.7e-6 + (38.25e-6 - 10.2e-6 + 6.0e-6) + 25.3e-6  #[kg*m^2]
inertia_yy = 92.7e-6 + (15.3e-6 - 3.07e-6 + 6.0e-6) + 25.3e-6  #[kg*m^2]
inertia_zz = 158.57e-6 + (38.25e-6 - 10.9e-6) + 25.3e-6 #[kg*m^2]
frictionCoeff = 0.2

motorParam = [minThrust, maxThrust, torqueFromForceFwd, torqueFromForceRev]
inertialParam = [mass, inertia_xx, inertia_yy, inertia_zz]
myTensegrity = Tensegrity(rodLength,propX,propY,propRadius)
myTensegrityRot = TensegrityRotation(myTensegrity,motorParam,inertialParam)
myTensegrityPlotter = TensegrityPlotter(myTensegrity,myTensegrityRot)
myReorientationPlanner = Reorientation(myTensegrity, motorParam, inertialParam, frictionCoeff)

face0 = 12
face1 = 10
deltaAlpha = 1e-3 #angAcc at 0.01 rad/s^2

# Solve the rotation feasibility problem
[solutionIsConsistent, thrusts, r0, r1, node0Pos, node1Pos] = myReorientationPlanner.solve_rotation_problem(face0,face1,deltaAlpha)

# Plot the rotation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rot = Rotation.identity()

ax = myTensegrityPlotter.draw_tensegrity(rot,ax)
for i in range(20):
    myTensegrityPlotter.draw_face(i,rot,ax)
myTensegrityPlotter.draw_rotation(face0, face1, rot, ax) # Plot the rotation axis as a black arrow
myTensegrityPlotter.draw_thrust(rot,ax,thrusts) # Plot the thrust as red arrows
myTensegrityPlotter.draw_reactionForces(rot, ax, r0, r1, node0Pos, node1Pos) # Plot the reaction force from ground as blue arrows

# Set labels and title
ax.set_xlabel('Body_X')
ax.set_ylabel('Body_Y')
ax.set_zlabel('Body_Z')
ax.set_title('Rotation analysis')

plt.show()