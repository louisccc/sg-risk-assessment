# dataset_preparation
In this folder of the project, we record various lane change scenarios using CARLA and scenario runner. 

**CARLA** 
This folder contains the files to run the CARLA simulation. 

**Scenario_Runner**
This project created multiple traffic scenarios to simulate real world driving for CARLA. Scenarios include leading vehicles or pedestrians crossing. We used these scenarios while randomly choosing a vehicle to lane change.

**spawn_npc.py**
This file spawns a number of vehicles and pedestrians in the CARLA world. This is the main file we run to record lane changes. The recordings are saved into a folder called _out. 
- Each recording has 3 folders: raw_images, scene_raw, ss_images
 - **raw_images:** frames of the recording using rgb camera
 - **scene_raw:** information on the environment, cars, pedestrians in dictionary format
 - **ss_images:** frames of the recording using segmented segmentation camera

**sensors.py**
This file contains cameras and other sensors that attach to the ego vehicle. The camera is used to record lane changes. In addition, we created functions to extract information from the vehicle.

**lane_change_recorder.py**
This file chooses a random vehicle to lane change. It records each lane change and gathers information from the surrounding environment. 

# How to Execute
1. Download CARLA 0.9.8 - refer to https://carla.readthedocs.io/en/0.9.8/ to set up
2. Navigate to /scenario_runner and pip install -r requirements.txt
3. Run the CARLA executable
4. Run spawn_npc.py in synchronous mode to record lane changes
- ex. python spawn_npc.py -n 100 --sync --safe
