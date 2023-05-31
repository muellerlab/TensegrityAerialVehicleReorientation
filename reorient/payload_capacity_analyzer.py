import numpy as np
from py3dmath.py3dmath import Vec3, Rotation
from tensegrity.tensegrity import Tensegrity
from tensegrity.tensegrity_rotation import TensegrityRotation
import cvxpy as cvxpy
import pydot as pydot

"""
Analyzing the payload capacity of re-orientation
The class has three major abilities:
1. Compute what is the maximum point-mass payload that can be added while keeping rotation feasible. 
2. Plot the analysis graph.
"""
class ReorientationPayloadAnalyzer():
    def __init__(self, tensegrity:Tensegrity, motorParam, inertialParam, frictionCoeff):
        self.tensegrity = tensegrity
        self.tensegrityRotation = TensegrityRotation(tensegrity, motorParam, inertialParam)
        self.frictionCoeff = frictionCoeff
        self.biDir= True  #Allow propellers to rotate backwards.
        self.forceMarginTable = None #Table storing the margin b
        self.takeoffFaceIDs = [4,9]
        return
    
    def solve_rotation_payload_problem(self, face0, face1, zeroSumThrustFlag = False):
        """
        Check the rotation feasibility between two neighboring faces and compute the maximum 
        point mass payload that can be added to the COM while still making the rotaiton feasible. 

        Criteria: there exists a group of thrusts, within the thrust range
        that can offset the torque of gravity without sliding.

        This function setups the rotation feasibility problem with cvxpy and solve it.

        Inputs:
        face0: starting face
        face1: ending face
        zeroSumThrustFlag: if adding the additional constraint of making sum of thrust zero

        Returns:
        [solutionFound, thrusts, r0, r1, node0_b, node1_b, maxPayload]
        """
        axis, angle, rot_pt, isEdge = self.tensegrityRotation.get_rotation_axis_angle_point(face0, face1)
        COM = self.tensegrityRotation.COM
        # initial guess of thrusts
        maxThrust = self.tensegrityRotation.maxThrust 
        minThrust = self.tensegrityRotation.minThrust 
        if axis is None:
            # Two faces are not neighboring. No rotation    
            return [None, None, None, None, None, None, None]
        
        # Get common node point
        [nodes, rods, strings, motors, faces]= self.tensegrity.get_tensegrity_definition()
        mass = self.tensegrityRotation.mass
        thrustDirection = np.array([1, 1, 1, 1])  # Initial guess, thrusts are all positive.
        nf0 = faces[face0]
        nf1 = faces[face1]

        commonNodeIDs = []
        for i in range(3):
            if nf0[i] in nf1:
                commonNodeIDs += [nf0[i]]
        node0_b = Vec3(nodes[commonNodeIDs[0]]) 
        node1_b = Vec3(nodes[commonNodeIDs[1]])

        # Vector pointing from rotational point to node1
        arm0Matrix = (node0_b-rot_pt).to_cross_product_matrix()
        arm1Matrix = (node1_b-rot_pt).to_cross_product_matrix()
        armWeightMatrix = (COM-rot_pt).to_cross_product_matrix()
        normal_b = self.tensegrity.get_face_normal(face0)
        gravDir_b = -normal_b  # in body-fixed frame, where g is pointing
        weight_b = gravDir_b * 9.81 * mass
        momentWeight = (COM-rot_pt).cross(weight_b) #rot_pt is a vector from COM to the rotation point
        mass = self.tensegrityRotation.mass

        bestPayload = 0
        bestSolution = [False, None, None, None, node0_b, node1_b, None]

        if (self.biDir):
            # Go through all 16 possible combinations of thrust directions if we allow reverse spin.
            for i in range(16):
                thrustDirection = [int(x)*2-1 for x in list('{0:04b}'.format(i))]
                solutionFound = False
                M = self.tensegrityRotation.get_mixer_matrix(face0,face1,thrustDirection)
                [prob, thrusts, r0, r1, maxPayload] = self.solve_max_feasible_rotation_payload(thrustDirection, M, arm0Matrix, arm1Matrix, armWeightMatrix, normal_b, weight_b, momentWeight, zeroSumThrustFlag)
                if not prob.status in ["infeasible", "unbounded","infeasible_inaccurate"]:
                    solutionFound = True
                    if bestPayload is None:
                        print("somehtingWrong!!")
                    if maxPayload.value is None:
                        print("somehtingWrong!")
                    if maxPayload.value > bestPayload:
                        bestPayload = maxPayload.value
                        bestSolution = [solutionFound, thrusts.value, r0.value, r1.value, node0_b, node1_b, bestPayload]

            return bestSolution
        else:
            thrustDirection = [1,1,1,1]
            solutionFound = False
            M = self.tensegrityRotation.get_mixer_matrix(face0,face1,thrustDirection)
            # compute weight and moment due to weight:
            [prob, thrusts, r0, r1, maxPayload] = self.solve_max_feasible_rotation_payload(thrustDirection, M, arm0Matrix, arm1Matrix, armWeightMatrix, normal_b, weight_b, momentWeight, zeroSumThrustFlag)
            if not prob.status in ["infeasible", "unbounded"]:
                solutionFound = True
                return [solutionFound, thrusts.value, r0.value, r1.value, node0_b, node1_b, maxPayload]
            return[solutionFound, None, None, None, node0_b, node1_b, None]       


    def solve_max_feasible_rotation_payload(self, thrustDir, M, arm0Matrix:np.array, arm1Matrix:np.array, armWeightMatrix:np.array, normal_b:Vec3, weight_b:Vec3, momentWeight:Vec3, zeroSumThrustFlag = False):
        """
        Solve for the maximum payload we can add to the vehicle before we break the feasibility of re-orientation.
        When zeroSumThrustFlag = true, we enable the additional constraint of sum of thrust == 0 and disable the constraint of no-sliding
        When zeroSumThrustFlag = false, we disable the additional constraint of sum of thrust == 0 and endable the constraint of no-sliding
        """
        g = 9.81 # gravity constant

        sumForceMatrix = np.zeros([3, 4])
        sumForceMatrix[2,:] = [1,1,1,1]

        lowerThrustBound = np.zeros(4)
        upperThrustBound = np.zeros(4)

        maxThrust = self.tensegrityRotation.maxThrust 
        minThrust = self.tensegrityRotation.minThrust 

        for i in range(4):
            if thrustDir[i]>0:
                lowerThrustBound[i] = 0
                upperThrustBound[i] = maxThrust
            else:
                lowerThrustBound[i] = minThrust
                upperThrustBound[i] = 0            

        # 11 opt variables: 4 variables for propeller thrusts 
        # 6 variables for 2 ground reaction forces
        # 1 variable for maximum feasible payload (kg)
        # We analyze the problem about the rotation point

        optVar = cvxpy.Variable((1,11)) 
        thrusts = optVar[0,0:4]
        r0 = optVar[0,4:7] # ground reaction force0
        r1 = optVar[0,7:10] # ground reaction force1
        payloadMass = optVar[0,10] # the additional payload we add at center of mass.

        r0NormalValue = ((normal_b.to_array()).T @ r0 ) 
        r0NormalVector = r0NormalValue * normal_b.to_h_array()
        r0TangentVector = r0 - r0NormalVector
        r0TangentValue = cvxpy.norm(r0TangentVector,2)

        r1NormalValue = ((normal_b.to_array()).T @ r1 ) 
        r1NormalVector = r1NormalValue * normal_b.to_h_array()
        r1TangentVector = r1 - r1NormalVector
        r1TangentValue = cvxpy.norm(r1TangentVector,2)

        payloadWeightVector = -payloadMass * g * normal_b.to_h_array()
        payloadTorque = armWeightMatrix @ payloadWeightVector

        constraints = [
            r0 + r1 + weight_b.to_h_array() + sumForceMatrix @ thrusts + payloadWeightVector == np.array([0,0,0]), # Force balance WRT body frame
            arm0Matrix @ r0 + arm1Matrix @ r1 + M @ thrusts + momentWeight.to_h_array() + payloadTorque == np.array([0,0,0]),   # Moment balance
            thrusts >= lowerThrustBound, # Thrust upper bound
            thrusts <= upperThrustBound, # Thrust lower bound
            r0NormalValue >= 0, # node 0 in contact with ground
            r1NormalValue >= 0, # node 1 in contact with ground
            r0TangentValue <= self.frictionCoeff*r0NormalValue,
            r1TangentValue <= self.frictionCoeff*r1NormalValue,
            payloadMass >= 0] # payload mass is positive
        
        if zeroSumThrustFlag:
            constraints.append(cvxpy.sum(thrusts) == 0) #Sum of thrust == 0
        
        prob = cvxpy.Problem(cvxpy.Maximize(payloadMass),constraints)
        prob.solve(solver='SCS')
        return [prob, thrusts, r0, r1, payloadMass]

    def get_rotation_force_payload_table(self,face0, face1):
        [solutionExist, thrusts, r0, r1, node0, node1] = self.solve_rotation_problem(face0,face1)
        # Return None if rotation is infeasible
        if solutionExist is None:
            return None
        if not solutionExist:
            return None
        
        maxThrust = self.tensegrityRotation.maxThrust 
        minThrust = self.tensegrityRotation.minThrust 

        marginToMax = (maxThrust - np.max(thrusts)) / maxThrust  # how much room to max force is left
        marginToMin = (minThrust - np.min(thrusts)) / minThrust  # how much room to min force is left (positive means possible)
        
        return min([marginToMax, marginToMin])

    def get_rotation_force_payload_capacity_table(self, zeroSumThrustFlag = False):
        out = np.zeros([20, 20])
        for face0 in range(20):
            for face1 in range(20):
                [solutionFound, thrusts, r0, r1, node0_b, node1_b, maxPayload] = self.solve_rotation_payload_problem(face0, face1, zeroSumThrustFlag)
                if maxPayload is None:
                    out[face0, face1] = 0
                out[face0, face1] = maxPayload           
        return out
    
    def create_payload_capacity_map(self):
        graphFeasibleOnly = pydot.Dot("Tensegrity rotations", graph_type="digraph")
        payloadTable_ZeroSum = self.get_rotation_force_payload_capacity_table(zeroSumThrustFlag = True)
        payloadTable_NoConstraint = self.get_rotation_force_payload_capacity_table(zeroSumThrustFlag = False)

        # create all nodes:
        for face0 in range(20):
            if face0 in self.takeoffFaceIDs:
                graphFeasibleOnly.add_node(pydot.Node(face0, label=str(face0), color='blue', shape='square'))
            else:
                graphFeasibleOnly.add_node(pydot.Node(face0, label=str(face0)))
        # create edges:
        for face0 in range(20): 
            for face1 in range(20):
                payload_0f = payloadTable_ZeroSum[face0, face1]
                payload_no_0f = payloadTable_NoConstraint[face0, face1]
                if (payload_0f + payload_no_0f) > 0: 
                    l = "("+str(np.round(payload_0f, 3))+"," + str(np.round(payload_no_0f, 3)) + ")"
                    graphFeasibleOnly.add_edge(pydot.Edge(face0, face1, color='black', label=l))
        graphFeasibleOnly.write_png("graph_Payload.png")
        return