import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from py3dmath.py3dmath import Vec3,Rotation
import matplotlib.pyplot as plt

from tensegrity.tensegrity_plotter import TensegrityPlotter
from tensegrity.tensegrity import Tensegrity
from tensegrity.tensegrity_rotation import TensegrityRotation

import scipy.optimize
from matplotlib import rcParams


########################################
########################################
########################################

"""
This script compares the thrust conversion methods in terms of the norm of torque error generated.
1. Assume total thrust is 0. Do inverse with the full-size mixer matrix then do thrust saturate.
2. No total thrust requirement. Pseudo-inverse + thrust saturation.
3. No total thrust requirement. Use optimization method.

"""
firstTime = True # Run a new round of computation and save pickle file if the program is run first time 
rodLength = 0.20 #Length of tensegrity rod
propX = 0.05 #Offset x-distance for propeller 
propY = 0.034 #Offset y-distance for propeller
torqueFromForceFwd = 0.009975  # [m]
torqueFromForceRev = 0.0163  # [m] 
maxThrust = 2.8  # [N] 
minThrust = -1.5  # [N]
propRadius =  0.032 #[m] 2.5Inch prop
mass = 0.330  # [kg]
frictionCoeff = 0.2

inertia_xx = 7.82e-4 #[kg*m^2]
inertia_yy = 12.59e-4 #[kg*m^2]
inertia_zz = 12.90e-4 #[kg*m^2]
COMOffset = Vec3(0,0,0.01)

motorParam = [minThrust, maxThrust, torqueFromForceFwd, torqueFromForceRev]
inertialParam = [mass, inertia_xx, inertia_yy, inertia_zz]
myTensegrity = Tensegrity(rodLength,propX,propY,propRadius)
myTensegrityRot = TensegrityRotation(myTensegrity,motorParam,inertialParam)

face0=3 # start face
face1=4 # desired face to rotate to
ax0, angle, unitRotPt, isEdge = myTensegrityRot.get_rotation_axis_angle_point(face0, face1, unitFlag = True)
ax1 = unitRotPt.to_unit_vector()

tauAxis0Min = tauAxis1Min = -0.2
tauAxis0Max = tauAxis1Max = 0.2
secNumAxis0 = secNumAxis1 = 100

tau0 = np.linspace(tauAxis0Min, tauAxis0Max, secNumAxis0)
tau1 = np.linspace(tauAxis1Min, tauAxis1Max, secNumAxis1)

print(tau0.shape[0])
print(tau1.shape[0])
invErrorRateData = np.zeros([tau0.shape[0],tau1.shape[0]])
pinvErrorRateData = np.zeros([tau0.shape[0],tau1.shape[0]])
optErrorRateData = np.zeros([tau0.shape[0],tau1.shape[0]])
X, Y = np.meshgrid(tau0, tau1)


def thrustNorm(thrust):
    return np.linalg.norm(thrust)

def torqueError(thrust, M, tauCmd):
    # M 3x4 matrix
    # tauCmd 3x1 vector
    # thrust, unkown 1x4, will be reshaped to 4x1 
    tauOut = M @ thrust
    return np.linalg.norm(tauOut - tauCmd)

def torqueEq(thrust,M,tauCmd):
    tauOut = M @ thrust
    return tauOut - tauCmd

if firstTime:
    for i in range(tau0.shape[0]):
        for j in range (tau1.shape[0]):
            print("round=",(i,j))
            tauCmd = (tau0[i]*ax0+tau1[j]*ax1).to_array().squeeze()

            ## Part0: Inverse with zero total thrust assumption + saturation
            invErrorRate_ij = 100 # Initialize with a large error
            for k in range(16):
                fDirGuess = [int(x)*2-1 for x in list('{0:04b}'.format(k))]
                M_full = myTensegrityRot.get_full_mixer_matrix(face0,face1,fDirGuess)
                RHS = np.zeros(4)
                RHS[0:3] = tauCmd
                fCmd = np.linalg.inv(M_full) @ RHS # Compute the full
                #Saturate
                for l in range(4):
                    if fCmd[l] < minThrust:
                        fCmd[l] = minThrust
                    if fCmd[l] > maxThrust:
                        fCmd[l] = maxThrust

                M = myTensegrityRot.get_mixer_matrix(face0,face1,fDirGuess)
                invErrorRate = np.linalg.norm(M @ fCmd - tauCmd)/np.linalg.norm(tauCmd)
                if invErrorRate < invErrorRate_ij:
                    invErrorRate_ij = invErrorRate
                if invErrorRate < 1e-3:
                    invErrorRate_ij = 0
                    break
            invErrorRateData[i,j] = invErrorRate_ij

            ## Part1: Pseudoinverse + saturation
            pinvErrorRate_ij = 100 # Initialize with a large error
            for k in range(16):
                fDirGuess = [int(x)*2-1 for x in list('{0:04b}'.format(k))]
                M = myTensegrityRot.get_mixer_matrix(face0,face1,fDirGuess)
                fCmd = np.linalg.pinv(M) @ tauCmd #Compute the inverse
                for l in range(4):
                    if fCmd[l] < minThrust:
                        fCmd[l] = minThrust
                    if fCmd[l] > maxThrust:
                        fCmd[l] = maxThrust
                
                pinvErrorRate = np.linalg.norm(M @ fCmd - tauCmd)/np.linalg.norm(tauCmd)
                if pinvErrorRate < pinvErrorRate_ij:
                    pinvErrorRate_ij = pinvErrorRate
                if pinvErrorRate < 1e-3:
                    pinvErrorRate_ij = 0
                    break
            pinvErrorRateData[i,j] = pinvErrorRate_ij
            optErrorRate_ij = 100
            for k in range(16):
                fDirGuess = [int(x)*2-1 for x in list('{0:04b}'.format(k))]
                # Create the thrust bound
                bnds = []
                for l in range(4):
                    if fDirGuess[l]>=0:
                        lowerThrustBnd = 0
                        upperThrustBnd = maxThrust
                    else:
                        lowerThrustBnd = minThrust
                        upperThrustBnd = 0     
                    bnds.append((lowerThrustBnd,upperThrustBnd))
                
                M = myTensegrityRot.get_mixer_matrix(face0,face1,fDirGuess)
                f0 = np.zeros(4)
                cons = {'type': 'eq', 'fun': torqueEq, 'args': (M, tauCmd)} 
                res = scipy.optimize.minimize(thrustNorm, f0, constraints = cons, bounds = bnds)
                if res.success:
                    optErrorRate_ij = 0
                    break
                else:
                    f0 = np.linalg.pinv(M) @ tauCmd #Compute the inverse
                    res = scipy.optimize.minimize(torqueError, f0, method='SLSQP', bounds = bnds, args=(M, tauCmd))
                    fCmd = res.x
                    optErrorRate = np.linalg.norm(M @ fCmd - tauCmd)/np.linalg.norm(tauCmd)
                    if optErrorRate < optErrorRate_ij:
                        optErrorRate_ij = optErrorRate
            optErrorRateData[i,j] = optErrorRate_ij
    result =dict({"optErrorRate":optErrorRateData,"pinvErrorRate":pinvErrorRateData,"invErrorRate":invErrorRateData}) 
    file = open("optErrorRate"+".pickle", 'wb')
    pickle.dump(result, file)
    file.close()
else: 
    file = open("optErrorRate"+".pickle", 'rb')
    data = pickle.load(file)
    file.close()
    optErrorRateData = data["optErrorRate"]
    pinvErrorRateData = data["pinvErrorRate"]
    invErrorRateData = data["invErrorRate"]
# Creating figure 3d / heatmap
plotStyle = "heatmap" 
if plotStyle == "3d":
    rcParams['axes.labelpad'] = 50
    fig = plt.figure(figsize=(30, 20), dpi=100)
    labelSize = 40
    ax = plt.axes(projection="3d")
    ax.set_xlim(-0.15,0.15)
    ax.set_ylim(-0.15,0.15)
    ax.set_xlabel('Torque - Axis1 [Nm]',fontsize=labelSize, linespacing=100)
    ax.set_ylabel('Torque - Axis2 [Nm]',fontsize=labelSize, linespacing=100)
    ax.set_zlabel('Error Rate [Nm/Nm]',fontsize=labelSize, linespacing=100)
    surf = ax.plot_surface(X, Y, optErrorRateData, cmap='viridis',
                        linewidth=0, antialiased=False,shade=False)
    surf = ax.plot_surface(X, Y, pinvErrorRateData, cmap='plasma',
                        linewidth=0, antialiased=False,shade=False)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=labelSize)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='z', which='major', pad=20)
    ax.view_init(10, -45)
    plt.locator_params(nbins=5)  
    plt.savefig("comparisonResult.svg", format = 'svg', dpi=100)
    plt.savefig("comparisonResult.png", format = 'png', dpi=100)
    plt.show()

elif plotStyle == "heatmap":
    vmin = min([np.min(invErrorRateData),np.min(pinvErrorRateData), np.min(optErrorRateData)])
    vmax = max([np.max(invErrorRateData),np.max(pinvErrorRateData), np.max(optErrorRateData)])
    levels = np.linspace(0, 1.1, 12)
    annotate_levels = levels[::2]  # Select every second level for annotation

    fig1, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 5), sharex='col', sharey='row')
    (ax0, ax1, ax2) = axs

    fig1.suptitle('Command Error Rate', fontsize=10)

    cs0 = ax0.contour(X, Y, invErrorRateData, levels=levels, colors='black')
    cs0c =ax0.contourf(X, Y, invErrorRateData, levels=levels, cmap='BuPu')
    plt.clabel(cs0, levels = annotate_levels, inline=True, fontsize=8)  # annotate contours

    ax0.set_ylabel('Torque - Axis-2[Nm]', fontsize=10)
    ax0.set_xlabel('Torque - Axis-1[Nm]', fontsize=10)
    ax0.set_yticks(np.arange(-0.2, 0.21, 0.1))
    ax0.tick_params(labelsize=8)
    ax0.set_title('0-Sum + Inverse + Saturation', fontsize=10)
    ax0.set_aspect('equal')

    cs1 = ax1.contour(X, Y, pinvErrorRateData, levels=levels, colors='black')
    cs1c =ax1.contourf(X, Y, pinvErrorRateData, cmap='BuPu', levels=levels)
    plt.clabel(cs1, levels = annotate_levels, inline=True, fontsize=8)  # annotate contours
    ax1.set_xlabel('Torque - Axis-1[Nm]', fontsize=10)
    ax1.tick_params(labelsize=8)
    ax1.set_yticks(np.arange(-0.2, 0.21, 0.1))
    ax1.set_title('Pseudoinverse + Saturation', fontsize=10)
    ax1.set_aspect('equal')

    cs2 = ax2.contour(X, Y, optErrorRateData, levels=levels, colors='black')
    cs2c = ax2.contourf(X, Y, optErrorRateData, cmap='BuPu', levels=levels)
    plt.clabel(cs2, levels = annotate_levels, inline=True, fontsize=8)  # annotate contours
    ax2.set_xlabel('Torque - Axis-1[Nm]', fontsize=10)
    ax2.set_yticks(np.arange(-0.2, 0.21, 0.1))
    ax2.tick_params(labelsize=8)
    ax2.set_title('Optimization', fontsize=10)
    ax2.set_aspect('equal')
    
    cbar = fig1.colorbar(cs2c, ax=axs.ravel().tolist(), orientation='horizontal')
    cbar.set_ticks(levels)
    cbar.ax.tick_params(labelsize=8)  # Change the tick size on the colorbar

    plt.savefig("ThrustConverterComparison.pdf", format='pdf')  # save as pdf
    plt.savefig("ThrustConverterComparison.png", format='png', dpi=300)  # save as png
    plt.show()
print("done")
