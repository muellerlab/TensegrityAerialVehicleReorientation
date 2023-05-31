import numpy as np
from py3dmath.py3dmath import Vec3, Rotation
from tensegrity.tensegrity import Tensegrity
from tensegrity.tensegrity_rotation import TensegrityRotation
import cvxpy as cvxpy
import pydot as pydot

"""
Planner of tensegrity reorientation
The class has three major abilities:
1. Check if given tensegrity rotation between two faces is feasible.
2. Create the connected graph of possible reorientations. 
3. Search the graph for re-orientation routes. 
"""

class Reorientation():
    def __init__(self, tensegrity:Tensegrity, motorParam, inertialParam, frictionCoeff):
        self.tensegrity = tensegrity
        self.tensegrityRotation = TensegrityRotation(tensegrity, motorParam, inertialParam)
        self.frictionCoeff = frictionCoeff
        self.biDir= True  #Allow propellers to rotate backwards.
        self.forceMarginTable = None #Table storing the margin b
        self.takeoffFaceIDs = [4,9]
        return

    def solve_rotation_problem(self, face0, face1, deltaAlpha = 0):
        """
        Check if a rotation between two neighboring faces is feasible. 
        Criteria: there exists a group of thrusts, within the thrust range
        that can offset the torque of gravity without sliding.

        This function setups the rotation feasibility problem with cvxpy and solve it.

        Inputs:
        face0: starting face
        face1: ending face
        deltaAlpha: desired angular acceleration about rotation axis

        Returns:
        [solutionFound, thrusts, r0, r1, node0_b, node1_b]
        solutionFound: if the cvxpy finds a set of forces that meet the requirement. 
        thrusts: the set of thrusts which can offset the 
        """

        axis, angle, rot_pt, isEdge = self.tensegrityRotation.get_rotation_axis_angle_point(face0, face1)
        COM = self.tensegrityRotation.COM
        # initial guess of thrusts
        maxThrust = self.tensegrityRotation.maxThrust 
        minThrust = self.tensegrityRotation.minThrust 
        if axis is None:
            # Two faces are not neighboring. No rotation    
            return [None, None, None, None, None, None]
        
        # Get common node point
        [nodes, rods, strings, motors, faces]= self.tensegrity.get_tensegrity_definition()
        mass = self.tensegrityRotation.mass
        thrustDirection = np.array([1, 1, 1, 1])  #Initial guess, thrusts are all positive.
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
        normal_b = self.tensegrity.get_face_normal(face0)
        gravDir_b = -normal_b  # in body-fixed frame, where g is pointing
        weight_b = gravDir_b * 9.81 * mass
        momentWeight = (COM-rot_pt).cross(weight_b) #rot_pt is a vector from COM to the rotation point

        mass = self.tensegrityRotation.mass
        Ir_B = self.tensegrityRotation.get_inertial_matrix_about_rot_pt(face0,face1)
        
        desAlpha_B = deltaAlpha*axis
        desAccCOM_B = desAlpha_B.cross(COM-rot_pt) #Here we assume zero angular velocity

        desForce_B = mass * (desAccCOM_B.to_h_array())
        desTorque_B = Ir_B @ (desAlpha_B.to_h_array())

        if (self.biDir):
            # Go through all 16 possible combinations of thrust directions if we allow reverse spin.
            for i in range(16):
                thrustDirection = [int(x)*2-1 for x in list('{0:04b}'.format(i))]
                solutionFound = False
                M = self.tensegrityRotation.get_mixer_matrix(face0,face1,thrustDirection)
                [prob, thrusts, r0, r1] = self.solve_rotation_feasibility_problem(thrustDirection, M, arm0Matrix, arm1Matrix,normal_b, weight_b, momentWeight, desForce_B, desTorque_B)
                if not prob.status in ["infeasible", "unbounded"]:
                    solutionFound = True
                    return [solutionFound, thrusts.value, r0.value, r1.value, node0_b, node1_b]
            return[solutionFound, None, None, None, node0_b, node1_b]            
        else:
            thrustDirection = [1,1,1,1]
            solutionFound = False
            M = self.tensegrityRotation.get_mixer_matrix(face0,face1,thrustDirection)
            # compute weight and moment due to weight:
            [prob, thrusts, r0, r1] = self.solve_rotation_feasibility_problem(thrustDirection, M, arm0Matrix, arm1Matrix,normal_b, weight_b, momentWeight, desForce_B, desTorque_B)
            if not prob.status in ["infeasible", "unbounded"]:
                solutionFound = True
                return [solutionFound, thrusts.value, r0.value, r1.value, node0_b, node1_b]
            return[solutionFound, None, None, None, node0_b, node1_b]       

    def solve_rotation_feasibility_problem(self, thrustDir, M, arm0Matrix, arm1Matrix, normal_b:Vec3, weight_b:Vec3, momentWeight:Vec3, desForce_B:np.array, desTorque_B:np.array):
        """
        Setup and solve the rotation feasibility problem with cvx solver. 
        """
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

        thrustsMid = np.ones([4,]) * ((minThrust + maxThrust)/2) 

        # 10 opt variables: 4 variables for propeller thrusts 
        # 6 variables for 2 ground reaction forces
        # We analyze the problem about the rotation point
        optVar = cvxpy.Variable((1,10)) 
        thrusts = optVar[0,0:4]
        r0 = optVar[0,4:7] # ground reaction force0
        r1 = optVar[0,7:10] # ground reaction force1

        r0NormalValue = ((normal_b.to_array()).T @ r0 ) 
        r0NormalVector = r0NormalValue * normal_b.to_list()
        r0TangentVector = r0 - r0NormalVector
        r0TangentValue = cvxpy.norm(r0TangentVector,2)

        r1NormalValue = ((normal_b.to_array()).T @ r1 ) 
        r1NormalVector = r1NormalValue * normal_b.to_list()
        r1TangentVector = r1 - r1NormalVector
        r1TangentValue = cvxpy.norm(r1TangentVector,2)

        constraints = [
            r0 + r1 + weight_b.to_list() + sumForceMatrix @ thrusts == desForce_B, # Force balance WRT body frame
            arm0Matrix @ r0 + arm1Matrix @ r1 + M @ thrusts + momentWeight.to_list() == desTorque_B,   # Moment balance
            thrusts >= lowerThrustBound, # Thrust upper bound
            thrusts <= upperThrustBound, # Thrust lower bound
            r0NormalValue >= 0, # node 0 in contact with ground
            r1NormalValue >= 0, # node 1 in contact with ground
            r0TangentValue <= self.frictionCoeff*r0NormalValue, # node 0 non-slipping condition
            r1TangentValue <= self.frictionCoeff*r1NormalValue] # node 1 non-slipping condition]
        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.norm(thrusts-thrustsMid, np.inf)),constraints)
        prob.solve(solver='SCS')
        return [prob, thrusts, r0, r1]
    
    def get_rotation_force_reserve_fraction(self,face0, face1):
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

    def get_rotation_force_reserve_fraction_table(self):
        out = np.zeros([20, 20])
        for f0 in range(20):
            for f1 in range(20):
                v = self.get_rotation_force_reserve_fraction(f0, f1)
                if v is None:
                    v = 0
                out[f0, f1] = v            
        return out
    
    def create_feasibility_map(self):
        graphAllConnections = pydot.Dot("Tensegrity rotations", graph_type="digraph")
        graphFeasibleOnly = pydot.Dot("Tensegrity rotations", graph_type="digraph")
        neighborEdge, neighborPoint = self.tensegrity.get_face_neighbours()
        forceMarginTable = self.get_rotation_force_reserve_fraction_table()
        
        # create all nodes:
        for face0 in range(20):
            if face0 in self.takeoffFaceIDs:
                graphAllConnections.add_node(pydot.Node(face0, label=str(face0), color='blue', shape='square'))
                graphFeasibleOnly.add_node(pydot.Node(face0, label=str(face0), color='blue', shape='square'))
            else:
                graphAllConnections.add_node(pydot.Node(face0, label=str(face0)))
                graphFeasibleOnly.add_node(pydot.Node(face0, label=str(face0)))
        # create edges:
        for face0 in range(20): 
            for face1 in range(20):
                forceMargin = forceMarginTable[face0, face1]
                if forceMargin > 0: 
                    # label edges, larger label value => lower thrust margin => harder reorientation
                    l = str(np.round(1 / forceMargin, 1))
                    if face1 in neighborEdge[face0]: 
                        # edge neighbor:
                        graphAllConnections.add_edge(pydot.Edge(face0, face1, color='black', label=l)) 
                        graphFeasibleOnly.add_edge(pydot.Edge(face0, face1, color='black', label=l))
                    else:
                        graphAllConnections.add_edge(pydot.Edge(face0, face1, color='blue', style='dashed', label=l)) 
                        graphFeasibleOnly.add_edge(pydot.Edge(face0, face1, color='blue', style='dashed', label=l))
                else:
                    graphAllConnections.add_edge(pydot.Edge(face0, face1, color='red', style='dotted'))
        graphFeasibleOnly.write_png("graph_FeasibleOnly.png")
        graphAllConnections.write_png("graph_All.png")
        return
    
    def create_takeoff_map(self):
        # find path to takeoff face
        neighborEdge, neighborPoint = self.tensegrity.get_face_neighbours()
        if self.forceMarginTable is None:
            self.forceMarginTable = self.get_rotation_force_reserve_fraction_table()
        # nodesThatGoTo[f] is the list of faces that we can, in 1 step, reach f from
        nodesThatGoTo = {} 
        for faceTarget in range(20):
            nodesThatGoTo[faceTarget] = []
            for faceStart in range(20):
                forceMargin = self.forceMarginTable[faceStart, faceTarget]
                if forceMargin <= 0: 
                    # not reachable
                    continue
                if (not faceTarget in neighborEdge[faceStart]): 
                    continue
                nodesThatGoTo[faceTarget] += [faceStart]
                    
        shortestPathNextFace = np.zeros([20, ]) + np.inf
        shortestPathLength = np.zeros([20, ]) + np.inf

        for takeoffFaceID in self.takeoffFaceIDs:
            shortestPathNextFace[takeoffFaceID] = takeoffFaceID
            shortestPathLength[takeoffFaceID] = 0
        
        for s in range(20):
            for fEnd in range(20):
                if not shortestPathLength[fEnd] < np.inf:
                    # can't even reach fEnd yet
                    continue

                for fStart in nodesThatGoTo[fEnd]:
                    newCost = shortestPathLength[fEnd] + 1
                    
                    if newCost < shortestPathLength[fStart]:
                        shortestPathNextFace[fStart] = fEnd
                        shortestPathLength[fStart] = shortestPathLength[fEnd] + 1  # TODO better cost? 
        route = dict()

        # make the graph:
        graphToTakeoff = pydot.Dot("Shortest path to takeoff", graph_type="digraph")
        for face0 in range(20):
            if face0 in self.takeoffFaceIDs:
                graphToTakeoff.add_node(pydot.Node(face0, label=str(face0), color='blue', shape='square'))
            if np.isinf(shortestPathLength[face0]):
                # unreachable!
                graphToTakeoff.add_node(pydot.Node(face0, label=str(face0), color='red'))
            else:
                graphToTakeoff.add_node(pydot.Node(face0, label=str(face0)))
                
        for face in range(20):
            if face in self.takeoffFaceIDs:
                continue 
            
            if np.isinf(shortestPathNextFace[face]):
                # unconnected!
                continue

            nextFace = int(shortestPathNextFace[face]) 
            if nextFace in neighborEdge[face]:
                graphToTakeoff.add_edge(pydot.Edge(face, nextFace, color='black'))
                route[face] = nextFace
            else:
                graphToTakeoff.add_edge(pydot.Edge(face, nextFace, color='blue', style='dashed'))
        
        graphToTakeoff.write_png("graph_toTakeoff.png")
        return route



    """
    Additional analysis tools
    """

    def validate_rotation_solution(self, face0, face1, rotationSolution):
        """
        Given thrusts and reaction forces of a certain rotation, compute the total 
        thrust and torque. This function is used to validate the result of cvxpy solver 
        and analyze the numerical error, if there is one. 
        """
        [nodes, rods, strings, motors, faces]= self.tensegrity.get_tensegrity_definition()
        [solutionExist, thrusts, r0, r1, node0_b, node1_b] = rotationSolution
        axis, angle, rot_pt, isEdge = self.tensegrityRotation.get_rotation_axis_angle_point(face0, face1)
        COM = self.tensegrityRotation.COM
        mass = self.tensegrityRotation.mass
        motorThrustSign = np.sign(thrusts) #Sign list of the thrusts

        # Vector pointing from rotational point to node1
        arm0Matrix = (node0_b-rot_pt).to_cross_product_matrix()
        arm1Matrix = (node1_b-rot_pt).to_cross_product_matrix()

        # Mixer matrix
        M = self.tensegrityRotation.get_mixer_matrix(face0,face1,motorThrustSign)
        sumForceMatrix = np.zeros([3, 4])
        sumForceMatrix[2,:] = [1,1,1,1]

        # compute moment due to weight:
        normal_b = self.tensegrity.get_face_normal(face0)
        gravDir_b = -normal_b  # in body-fixed frame, where g is pointing
        weight_b = gravDir_b * 9.81 * mass
        momentWeight = (COM-rot_pt).cross(weight_b).to_list() #rot_pt is a vector from COM to the rotation point, so it equals the negative of vector from rotation point to COM, where the gravity force acts on.
        totalForce = r0 + r1 + weight_b.to_list() + sumForceMatrix @ thrusts
        totalTorque = arm0Matrix @ r0 + arm1Matrix @ r1 + M @ thrusts + momentWeight  

        if np.linalg.norm(totalForce) < 1e-3 and np.linalg.norm(totalTorque) < 1e-5:
            return [True, totalForce, totalTorque]
        else:
            return [False, totalForce, totalTorque]