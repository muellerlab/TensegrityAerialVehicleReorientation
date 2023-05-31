# Tensegrity Aerial Vehicle Re-orientation Analysis

This repository contains source code implementing the re-orientation feasibility analysis and route planning of a tensegrity aerial vehicle.

The detail of the simulation is described in the paper "Design and control of a collision-resilient aerial vehicle with an icosahedron tensegrity structure" submitted to  IEEE/ASME Transactions on Mechatronics (TMECH). A manuscript draft can be accessed [here](https://hiperlab.berkeley.edu/wp-content/uploads/2022/11/Design-and-control-of-a-collision-resilient-aerial-vehicle-with-an-icosahedron-tensegrity-structure.pdf). 

This work is evolved from our previous [IROS 2020 paper](https://ieeexplore.ieee.org/document/9341236).

Contact: Clark Zha (clark.zha@berkeley.edu)
High Performance Robotics Lab, Dept. of Mechanical Engineering, UC Berkeley

## Dependencies
The code uses following common python packages:
```
numpy, scipy, matplotlib, cvxpy, pydot
```

In addition, the code uses [py3dmath](https://github.com/muellerlab/TensegrityAerialVehicleCollisionSim) for 3D vector computation. For the ease of usage, we include a copy of the package in this repository so no additional installation is required.  

We also provide a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment file to help with the environment setup process. Simply run
```
conda env create -f environment.yml
```
with the provided yml file in your terminal to setup a proper python environment to run the code. 

## Quick Start Guide

To check the feasibility and plan the re-orientation of an example tensegrity aerial vehicle, run:
```
python reorientation_analysis.py
```

To visualize the thrusts and reaction forces during a tensegrity aerial vehicle rotation, run:
```
python plot_rotation.py
```

To recreate the torque converter analysis in the paper, run:
```
python torque_converter_test.py
```

To recreate the additional payload analysis in the paper, run:
```
python reorientation_payload_capacity.py
```


## Using this software as tensegrity design and analysis tool: 
To create and test your own tensegrity vehicle, you can modify parameters such as mass, size, moment of inertia, etc. The ```tensegrity``` folder contains the class object that keeps track of the tensegrity class. The ```reorient``` folder contains code that helps check the feasibility of rotation and plan the re-orientation route.


## Acknowledgement
Co-authors of the paper: Xiangyu Wu, Ryan Dimick, Mark. W. Mueller
Collaborators who have contributted to the tensegrity aerial vehicle developement: Joey Kroeger, Natalia Perez, Bryan Yang
Scholars who have provided their insights on the tensegrity aerial vehicle: Alice Agogino, Alan Zhang, Douglas Hutchings, Kévin Garanger
