# dataset_preparation
In this folder of the project, we record various lane change scenarios using CARLA and scenario runner. 

**CARLA** 
This folder contains the files to execute the CARLA simulation. In our project, we used manual_control.py to execute different lane change scenarios. 
manual_control.py can be found under the examples folder.

**Scenario_Runner**
This project created multiple traffic scenarios to simulate real world driving for CARLA. Scenarios include leading vehicles or pedestrians crossing. We used these scenarios while randomly choosing vehicle to lane change.

**spawn_npc.py**
This file spawns a number of vehicles and pedestrians in the CARLA world. 

**sensors.py**
This file contains cameras and other sensors that attach to the ego vehicle. The camera is used to record lane changes. In addition, we created functions to extract information from the vehicle.

**lane_change_recorder.py**
This file chooses a random vehicle to lane change. It records each lane change and gathers information from the surrounding environment. 

# How to Execute