"""
Tensegrity Plotter
"""
import numpy as np
from py3dmath.py3dmath import Vec3, Rotation
from tensegrity.tensegrity import Tensegrity
from tensegrity.tensegrity_rotation import TensegrityRotation

class TensegrityPlotter():
    def __init__(self,tensegrity:Tensegrity, tensegrityRotation:TensegrityRotation):
        self.tensegrity = tensegrity
        self.tensegrityRotation = tensegrityRotation

    def draw_tensegrity(self, rot, ax):
        """
        Draw the tensegrity structure
        """
        [nodes,rods,strings,motors,faces] = self.tensegrity.get_tensegrity_definition()

        # rods
        for b, e in rods:
            c1 = rot * nodes[b]
            c2 = rot * nodes[e]
            coords = np.array([c1.to_list(), c2.to_list()])
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', linewidth=3)

        # strings 
        for b, e in strings:
            c1 = rot * nodes[b]
            c2 = rot * nodes[e]
            coords = np.array([c1.to_list(), c2.to_list()])
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'r:')

        # propellers
        for i in range(4):
            pos = rot * motors[i]
            propAx = rot * Vec3(0, 0, 1)
            radius = self.tensegrity.propRadius
            if i == 0 or i == 2:
                col = 'g'
            else:
                col = 'r'
            self.draw_propeller(ax, pos, propAx, radius, col)
        return ax

    def draw_face(self, faceID, rot, ax, colorspec='c-'):
        """
        Paint color to tensegrity faces and add index.
        """
        ids = self.tensegrity.faces[faceID]
        nodes_b = self.tensegrity.nodes
        n0 = rot * nodes_b[ids[0]]
        n1 = rot * nodes_b[ids[1]]
        n2 = rot * nodes_b[ids[2]]
        coords = np.array([n0.to_list(), n1.to_list(), n2.to_list(), n0.to_list()])
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], colorspec, linewidth=2)
        c = (n0 + n1 + n2) / 3
        ax.text(c.x, c.y, c.z, str(faceID), color=colorspec[0])
        return 
    
    def draw_propeller(self, ax, center, normal, radius, col):
        """
        Paint color to tensegrity faces and add index.
        """
        nPoints = 50
        theta = np.linspace(0, 2 * np.pi, nPoints)
        
        xVals = np.zeros([nPoints, ])
        yVals = np.zeros([nPoints, ])
        zVals = np.zeros([nPoints, ])
        
        n0 = normal.to_unit_vector()
        e1 = Vec3(1, 0, 0)
        e2 = Vec3(0, 1, 0)
        if (e1.cross(n0)).norm2() > (e2.cross(n0)).norm2():
            n1 = (e1.cross(n0)).to_unit_vector()
        else:
            n1 = (e2.cross(n0)).to_unit_vector()
        
        n2 = n1.cross(n0)
        
        for i in range(nPoints):
            p = center + radius * (np.sin(theta[i]) * n2 + np.cos(theta[i]) * n1)
            xVals[i] = p.x
            yVals[i] = p.y
            zVals[i] = p.z
        
        ax.plot(xVals, yVals, zVals, col + '-')
        ax.plot([center.x], [center.y], [center.z], col + 'o')

    def draw_rotation(self,face_0, face_1, rot, ax):
        """
        Draw the rotation axis, shown as a black arrow.
        """
        axis, angle, pt, isEdge = self.tensegrityRotation.get_rotation_axis_angle_point(face_0, face_1)
        p0 = rot * pt
        p1 = rot * (pt + axis * (angle/np.pi)*self.tensegrity.rodLength) # Length of rotation axis represent the rotation angle.
        ax.quiver(p0[0],p0[1],p0[2],p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2], length = 20*(angle/(np.pi))*self.tensegrity.rodLength, linewidth = 3, color='k',arrow_length_ratio=0.4)
        pass

    def draw_thrust(self, rot, ax, thrustList):
        """
        Draw the thrusts
        """
        [nodes,rods,strings,motors,faces] = self.tensegrity.get_tensegrity_definition()
        quiverList = []
        forceScale = self.tensegrity.rodLength/5
        for i in range(4):
            pos = rot * motors[i]
            propAx =  rot * Vec3(0, 0, np.sign(thrustList[i]))
            quiverList.append(ax.quiver(pos[0],pos[1],pos[2], propAx[0], propAx[1], propAx[2], length = np.abs(thrustList[i])*forceScale, linewidth = 3, color='r',arrow_length_ratio=0.4))
        pass

    def draw_reactionForces(self,  rot, ax, r0, r1, node0, node1):
        """
        Draw the reaction force for rotation from face0 to face1
        """
        node0 = rot*Vec3(node0)
        r0 = rot*Vec3(r0)
        node1 = rot*Vec3(node1)
        r1 = rot*Vec3(r1)
        forceScale = self.tensegrity.rodLength/5
        ax.quiver(node0[0],node0[1],node0[2],r0[0],r0[1], r0[2], length = r0.norm2()*forceScale, linewidth = 3, arrow_length_ratio=0.4)
        ax.quiver(node1[0],node1[1],node1[2],r1[0],r1[1], r1[2], length = r1.norm2()*forceScale, linewidth = 3, arrow_length_ratio=0.4)
        pass 
