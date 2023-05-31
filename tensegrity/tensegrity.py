"""
The Tensegrity Aerial Vehicle Class is a class object that is designed 
to manage and keep track of the geometrical parameters of a tensegrity structure. 
It has the ability to manage various properties such as nodes, faces, normal directions, 
propeller positions, and other geometry-related properties. 
"""
import numpy as np
from py3dmath.py3dmath import Vec3,Rotation

class Tensegrity():
    def __init__(self, rodLength, propX, propY, propRadius) -> None:
        """
        Define the tensegrity aerial vehicle
        """
        self.dim = 3 # We consider a 3d problem
        self.numFace = 20
        self.numNode = 12
        self.numString = 24 
        self.numRod = 6 
        self.rodLength = rodLength
        # 12 tensegrity nodes + 4 mass nodes representing quadcopter weight
        unitTensegrityNodes = [
            [0.00, 0.50, 0.25], 
            [1.00, 0.50, 0.25], 
            [0.00, 0.50, 0.75], 
            [1.00, 0.50, 0.75], 
            [0.25, 0.00, 0.50], 
            [0.25, 1.00, 0.50], 
            [0.75, 0.00, 0.50], 
            [0.75, 1.00, 0.50], 
            [0.50, 0.25, 0.00], 
            [0.50, 0.25, 1.00], 
            [0.50, 0.75, 0.00], 
            [0.50, 0.75, 1.00]]
        
        self.unitNodePos = []
        for n in unitTensegrityNodes:
            nv = Vec3(n) - Vec3(0.5, 0.5, 0.5)
            self.unitNodePos += [nv]
        
        self.nodes = []
        for i in range(self.numNode):
            self.nodes.append(Vec3(self.unitNodePos[i])*self.rodLength)
       
        # string connectivity
        self.strings = [[0, 4], [0, 5], [0, 8], [0, 10], [1, 6], [1, 7], [1, 8], [1, 10], \
                        [2, 4], [2, 5], [2, 9], [2, 11], [3, 6], [3, 7], [3, 9], [3, 11], \
                        [4, 8], [4, 9], [5, 10], [5, 11], [6, 8], [6, 9], [7, 10], [7, 11]]
        
        # rod connectivity
        self.rods = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]  
        self.faces = self.get_faces()

        # positions of motors in body frame
        self.motors = []
        self.motors += [Vec3(+propX, -propY, 0)]
        self.motors += [Vec3(-propX, -propY, 0)]
        self.motors += [Vec3(-propX, +propY, 0)]
        self.motors += [Vec3(+propX, +propY, 0)]
        self.propRadius = propRadius

    def get_faces(self):
        # return enumerated list of faces, each face defined by the index of the three nodes making it up
        # each face is defined by three nodes:
        # by design, n0 < n1 < n2
        faces = []
        for i0 in range(self.numNode):
            for i1 in range(i0 + 1, self.numNode):
                for i2 in range(i1 + 1, self.numNode):
                    n0 = self.unitNodePos[i0]
                    n1 = self.unitNodePos[i1]
                    n2 = self.unitNodePos[i2]
                    # check if it makes a face:
                    c = (n0 + n1 + n2) / 3
                    # here we use geometric property, distance between center point 
                    # and the center of tensegrity is more than 0.4 of rod length.
                    if c.norm2() < 0.4:
                        continue
                    faces += [[i0, i1, i2]]
        return faces
    
    def get_face_normal(self,faceId)->Vec3:
        """
        get the normal vector of a face which points into the tensegrity
        """
        
        nid = self.faces[faceId]
        n0 = self.unitNodePos[nid[0]]
        n1 = self.unitNodePos[nid[1]]
        n2 = self.unitNodePos[nid[2]]
        
        # two vectors:
        v0 = n1 - n0
        v1 = n2 - n0
        
        normal = v0.cross(v1)
        normal = normal.to_unit_vector()
        
        # make sure it points inwards (n0 can be seen as vector from center to a node, i.e. outwards):
        if normal.dot(n0) > 0:
            normal = -normal
        return normal
    
    def get_tensegrity_definition(self):
        return [self.nodes, self.rods, self.strings, self.motors,self.faces]
    
    def get_face_neighbours(self):
        """
        Get the face ids for all neighboring faces, 
        returned as two dictionaries (share an edge, or share a point)
        """
        neighbors_edge = {}
        neighbors_point = {}
        for face_i in range(len(self.faces)):
            neighbors_edge[face_i] = []
            neighbors_point[face_i] = []
            for face_j in range(len(self.faces)):
                neighborCount = 0
                for k in range(3):
                    if self.faces[face_i][k] in self.faces[face_j]:
                        neighborCount += 1
                
                if neighborCount == 3:
                    # same face! 
                    continue
                if neighborCount == 2:
                    # share two nodes, i.e. share an edge
                    neighbors_edge[face_i] += [face_j]
                if neighborCount in [1, 2]:
                    # share at least one node (includes edges)
                    neighbors_point[face_i] += [face_j]
        return neighbors_edge, neighbors_point