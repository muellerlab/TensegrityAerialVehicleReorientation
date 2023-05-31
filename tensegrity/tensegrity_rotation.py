"""
Tensegrity Aerial Vehicle Rotation
The class keeps track of mass, moment of inertia, rotation axis and angle 
and other properties related to the rotation of tensegrity.
"""
import numpy as np
from py3dmath.py3dmath import Vec3,Rotation
from tensegrity.tensegrity import Tensegrity


class TensegrityRotation():
    def __init__(self,tensegrity:Tensegrity, motorParam, inertialParam, COMOffset = Vec3(0,0,0)) -> None:
        self.tensegrity = tensegrity
        [self.minThrust, self.maxThrust, self.torqueFromForceFwd, self.torqueFromForceRev] = motorParam
        [self.mass, self.Ixx, self.Iyy, self.Izz] = inertialParam
        self.COM = COMOffset # Position off set of center of mass, default to [0,0,0]

    def get_rotation_axis_angle_point(self, face0, face1, unitFlag = False):
        ''' 
        Analysis tool for how to rotate from face0 to face1
        Return 4 objects: axis, angle, point of contact of rotation. 
        When unitFlag is True, return the unit rotation point instead. 
        '''
        faces = self.tensegrity.faces
        if unitFlag:
            nodes_b = self.tensegrity.unitNodePos
        else:
            nodes_b = self.tensegrity.nodes
            
        nf0 = faces[face0]
        nf1 = faces[face1]

        commonNodeIDs = []
        for i in range(3):
            if nf0[i] in nf1:
                commonNodeIDs += [nf0[i]]

        if len(commonNodeIDs) == 2:
            # edge neighbor
            isEdge = True
            point = (nodes_b[commonNodeIDs[0]] + nodes_b[commonNodeIDs[1]]) / 2

        else:
            return None, None, None, False
            
        normal0 = self.tensegrity.get_face_normal(face0)
        normal1 = self.tensegrity.get_face_normal(face1)
        ax_sin_angle = normal0.cross(normal1)
        
        # should never be zero 
        ax = -ax_sin_angle * (1 / ax_sin_angle.norm2())  # NOTE sign! 
        angle = np.arcsin(ax_sin_angle.norm2())
        return ax, angle, point, isEdge
    
    def get_mixer_matrix(self, face0, face1, f):
        """
        Caluclate the mixer matrix that convert thrust into torque w.r.t to the 
        "rotation point", which is the mid point of the common edge of face0 and face1.
        f: thrust comands 
        When given a faceID<0, a mixer matrix w.r.t center of mass is returned.
        """
        if face0 >=0:
            axis, angle, rot_pt, isEdge = self.get_rotation_axis_angle_point(face0, face1)
        else:
            rot_pt = Vec3(0,0,0)
        motorPos = self.tensegrity.get_tensegrity_definition()[3]
        M = np.zeros([3, 4])  
        n = Vec3(0, 0, 1)
        for i in range(4):
            mi = motorPos[i]  # position of motor i in world frame
            r = mi-rot_pt  # displacement of force w.r.t. rotation point
            if f[i] >= 0:
                g = self.torqueFromForceFwd
            else:
                g = self.torqueFromForceRev
            v = ((-1) ** (i+1)) * g * n + r.cross(n) # (-1)**(i+1) here takes care of the handness of propeller.
            M[:, i] = v.to_list()
        return M


    def get_full_mixer_matrix(self, face0, face1, f):
        """
        Caluclate the "full" mixer matrix that convert 4 thrusts command into R^3 torque w.r.t to the 
        "rotation point", which is the mid point of the common edge of face0 and face1 + R^1 total thrusts.
        f: thrust comands 
        When given a faceID<0, a mixer matrix w.r.t center of mass is returned.
        """
        if face0 >=0:
            axis, angle, rot_pt, isEdge = self.get_rotation_axis_angle_point(face0, face1)
        else:
            rot_pt = Vec3(0,0,0)
        motorPos = self.tensegrity.get_tensegrity_definition()[3]
        M = np.zeros([4, 4])  
        n = Vec3(0, 0, 1)
        for i in range(4):
            mi = motorPos[i]  # position of motor i in world frame
            r = mi-rot_pt  # displacement of force w.r.t. rotation point
            if f[i] >= 0:
                g = self.torqueFromForceFwd
            else:
                g = self.torqueFromForceRev
            v = ((-1) ** (i+1)) * g * n + r.cross(n) # (-1)**(i+1) here takes care of the handness of propeller.
            M[0:3, i] = v.to_list()
        M[3,:] = np.array([1,1,1,1])
        return M


    def get_inertial_matrix_about_rot_pt(self,face0,face1):
        """
        Caluclate the inertial matrix with respect to the rotation point (in the body fixed frame).
        Here we use the generalized prallel axis theorem for 3D axis shift
        https://en.wikipedia.org/wiki/Parallel_axis_theorem
        """
        Ic = np.zeros([3,3])
        Ic[0,0] = self.Ixx
        Ic[1,1] = self.Iyy
        Ic[2,2] = self.Izz

        if face0 >=0:
            axis, angle, rot_pt, isEdge = self.get_rotation_axis_angle_point(face0, face1)
        else:
            return Ic
        
        dMatrix = (self.COM-rot_pt).to_cross_product_matrix() #skew-symmetric matrix of the vector from reference point to COM
        offSetMatrix = self.mass*(dMatrix @ dMatrix)

        return np.array(Ic - offSetMatrix)
    
    def get_face_reduced_attitude(self,faceID):
        """
        Get the reduced attitude (pitch and roll) when a face is contacting the ground. 
        The attitude is the rotation that rotates the face normal to a unit vector pointing up in world frame.
        """
        # normal vector of the face in body frame
        n = self.tensegrity.get_face_normal(faceID)
        # normal vector pointing up in world frame
        eWz = Vec3(0,0,1)

        # Compute the cross product and angle between the vectors
        v = n.cross(eWz)
        sinTheta = v.norm2()
        cosTheta = n.dot(eWz)

        # Compute the skew-symmetric cross product matrix
        v_matrix = np.array(v.to_cross_product_matrix())
        # Compute the rotation matrix
        if sinTheta != 0:
            R = np.identity(3) + v_matrix + np.matmul(v_matrix, v_matrix) * ((1 - cosTheta) / sinTheta**2)
        else:
            R = np.identity(3)
        att = Rotation.from_rotation_matrix(R)
        return att.to_euler_YPR()[1:3]
    
    def get_face_accelerometer_reading(self,faceID):
        """
        Get the accelerometer reading (i.e. proper acceleration) when the given face is on the ground.
        """
        g = 9.81 #gravity constant
        return g*self.tensegrity.get_face_normal(faceID)