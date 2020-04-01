import random
import sensors
import sys
from pathlib import Path

import os, math, json
from collections import defaultdict
import carla

from sensors import get_actor_attributes, get_vehicle_attributes
from agents.navigation.basic_agent import BasicAgent 
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import LaneChange

import py_trees

class LaneChangeRecorder:

    def __init__(self, traffic_manager, carla_world, client):
        # It has Idle, 
        self.state = "Idle"
        self.tick_count = 0
        self.carla_world = carla_world
        self.map = self.carla_world.get_map()
        self.vehicles_list = []
        self.traffic_manager = traffic_manager
        self.sensors_dict = {}
        self.root_path = Path("./_out")
        self.root_path.mkdir(exist_ok=True)

        self.num_of_existing_datapoints = len(list(self.root_path.glob('*')))
        self.dir_index = 0

        self.is_recording = False
        self.lane_changing= False

        self.agent = None
        self.client = client
       
    def set_vehicles_list(self, vehicles_list):
        self.vehicles_list = vehicles_list

    def attach_sensors(self, root_path):
        """
        Spawn and attach sensors to ego vehicles
        """
        cam_index = 0
        cam_pos_index = 1
        # dimensions = [1280, 720]
        dimensions = [640, 360]
        gamma = 2.2

        # self.sensors_dict["collision_sensor"] = sensors.CollisionSensor(self.ego)
        # self.sensors_dict["lane_invasion_sensor"] = sensors.LaneInvasionSensor(self.ego)
        # self.sensors_dict["gnss_sensor"] = sensors.GnssSensor(self.ego)
        self.sensors_dict["camera_manager"] = sensors.CameraManager(self.ego, gamma, dimensions, root_path)
        self.sensors_dict["camera_manager"].transform_index = cam_pos_index
        self.sensors_dict["camera_manager"].set_sensor(cam_index, notify=False)
        self.sensors_dict["camera_manager_ss"] = sensors.CameraManager(self.ego, gamma, dimensions, root_path)
        self.sensors_dict["camera_manager_ss"].transform_index = cam_pos_index
        self.sensors_dict["camera_manager_ss"].set_sensor(cam_index+5, notify=False)

    def destroy_sensors(self):
        for _, sensor in self.sensors_dict.items():
            sensor.destroy()
        self.sensors_dict = {}
        
    def toggle_recording(self):
        if 'camera_manager' in self.sensors_dict:
            self.sensors_dict["camera_manager"].toggle_recording()

        if 'camera_manager_ss' in self.sensors_dict:
            self.sensors_dict["camera_manager_ss"].toggle_recording()

    def tick(self, frame_num):
        self.tick_count += 1
        abs_velocity = lambda l: (3.6 * math.sqrt(l.x**2 + l.y**2 + l.z**2))
        
        if self.tick_count == 30:
            # choose random vehicle and prepare for recording
            print("Picking vehicle and attaching sensors...")
            self.ego = self.carla_world.get_actor(random.choice(self.vehicles_list))
            self.carla_world.get_spectator().set_transform(self.ego.get_transform())

            print("Attempting lane change...")
            self.lane_change_direction = None
            
            # check available lane changes
            waypoint = self.map.get_waypoint(self.ego.get_location())
            velocity = self.ego.get_velocity()
            scalar_velocity = int(abs_velocity(velocity))

            if ( waypoint.lane_change == carla.LaneChange.NONE or 
                (abs(velocity.x) <= 1.0 and abs(velocity.y) <= 1.0) or
                self.ego.is_at_traffic_light() ):
                print("Lane Change not available.")
                self.tick_count = 0
                return
            elif (waypoint.lane_change == carla.LaneChange.Both):
                print("Both")
                self.lane_change_direction = random.choice([True, False])
            elif (waypoint.lane_change == carla.LaneChange.Left):
                print("Left")
                self.lane_change_direction = True
            else:
                print("Right")
                self.lane_change_direction = False

            print("Start Lane Changing and Recording...")
            self.is_recording = True
            self.lane_changing= True 

            new_path = "%s/%s" % (str(self.root_path), self.num_of_existing_datapoints + self.dir_index)
            self.extractor = DataExtractor(self.ego, new_path)
            self.attach_sensors(new_path)
            self.dir_index += 1
            self.toggle_recording()

            ## setting 
            if self.lane_change_direction: # left lane change
                # wp_left = waypoint.get_left_lane()
                # next_wp = wp_left.next(25)[0]
                self.lane_change_controller = LaneChange(self.ego, direction='left')
            else:
                # wp_right = waypoint.get_right_lane()
                # next_wp = wp_right.next(25)[0]
                self.lane_change_controller = LaneChange(self.ego, direction='right')
            self.lane_change_controller.initialise()

            self.client.apply_batch_sync([carla.command.SetAutopilot(self.ego, False)], True)
            # if self.agent: ## avoid the vehicle being destroyed.
            #     self.agent._local_planner.reset_vehicle() 
            # self.agent = BasicAgent(self.ego, target_speed=scalar_velocity)
            # self.agent.set_destination((next_wp.transform.location.x, next_wp.transform.location.y, next_wp.transform.location.z))
              
        if self.lane_changing:
            success = self.lane_change_controller.update()
            if success == py_trees.common.Status.SUCCESS:
                self.lane_changing = False
                print('set set_autopilot back to true')
                self.client.apply_batch_sync([carla.command.SetAutopilot(self.ego, True)], True)
                self.toggle_recording()
                self.is_recording = False
                print("Cleaning up sensors...")
                self.destroy_sensors()
                self.tick_count = 0

        if self.is_recording:
            self.extractor.extract_frame(self.carla_world, self.map, frame_num)
            waypoint = self.map.get_waypoint(self.ego.get_location())
            velocity = self.ego.get_velocity()
            scalar_velocity = int(abs_velocity(velocity))
            print(scalar_velocity, waypoint.is_junction, waypoint.lane_type, waypoint.lane_change)

            # if self.lane_changing:
            #     if self.agent.done():
            #         next_wp = self.map.get_waypoint(self.ego.get_location()).next(scalar_velocity)[0]
            #         self.agent.set_destination((next_wp.transform.location.x, next_wp.transform.location.y, next_wp.transform.location.z))
            #         self.lane_changing = False
            # else:
            #     if self.agent.done():
            #         print('set set_autopilot back to true')
            #         self.client.apply_batch_sync([carla.command.SetAutopilot(self.ego, True)], True)
            #         self.toggle_recording()
            #         self.is_recording = False
            #         print("Cleaning up sensors...")
            #         self.destroy_sensors()

            #         self.tick_count = 0

        # if self.agent:
        #     control = self.agent.run_step(debug=True)
        #     control.manual_gear_shift = False
        #     self.ego.apply_control(control)


class DataExtractor(object):

    def __init__(self, ego, store_path):
        
        self.output_root_dir = Path(store_path).resolve()
        self.output_dir = (Path(store_path) / 'scene_raw').resolve()

        self.output_root_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.framedict=defaultdict()
        self.ego = ego

    def extract_frame(self, world, map1, frame):
        # utilities
        t = self.ego.get_transform()
        ego_location = self.ego.get_location()
        # velocity = lambda l: (3.6 * math.sqrt(l.x**2 + l.y**2 + l.z**2))
        # dv = lambda l: (3.6 * math.sqrt((l.x-v.x)**2 + (l.y-v.y)**2 + (l.z-v.z)**2))
        distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)

        vehicles = world.get_actors().filter('vehicle.*')
        pedestrians = world.get_actors().filter('walker.*')
        trafficlights = world.get_actors().filter('traffic.traffic_light')
        signs = world.get_actors().filter('traffic.traffic_sign')
        
        egodict = defaultdict()
        actordict = defaultdict()
        peddict = defaultdict()
        lightdict = defaultdict()
        signdict = defaultdict()
        lanedict = defaultdict()

        waypoint = map1.get_waypoint(ego_location, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))

        lanes = [("ego_lane", waypoint),
                 # ("next_waypoint", waypoint.next(10))
                 ]  #selecting a default 10 feet ahead for the next waypoint
        left_lanes = []
        right_lanes = [] 
        
        left_lane = waypoint 
        while True: 
            lane = left_lane.get_left_lane()
            if lane is None:
                break 
            left_lanes.append(("lane", lane))
            # print("left", lane.lane_type, lane.lane_change, lane.lane_id)
            if lane.lane_type in [carla.LaneType.Shoulder, carla.LaneType.Sidewalk]:
                break
            if left_lane.lane_id * lane.lane_id < 0: ## special handling.
                break
            left_lane = lane
            
        right_lane = waypoint
        while True:
            lane = right_lane.get_right_lane()
            if lane is None:
                break
            right_lanes.append(("lane", lane))
            # print("right", lane.lane_type, lane.lane_change, lane.lane_id)


            if lane.lane_type in [carla.LaneType.Shoulder, carla.LaneType.Sidewalk]:
                break
            if right_lane.lane_id * lane.lane_id < 0:
                break
            right_lane = lane
            
        lanes = left_lanes[::-1] + lanes + right_lanes
        
        lanedict['left_lanes'] = []
        for name, lane in left_lanes:
            single_lane_dict = {
                'lane_id': lane.lane_id,
                'lane_type': waypoint.lane_type, 
                'lane_width': waypoint.lane_width, 
                'right_lane_color': waypoint.right_lane_marking.color, 
                'left_lane_color': waypoint.left_lane_marking.color,
                'right_lane_marking_type': waypoint.right_lane_marking.type, 
                'left_lane_marking_type': waypoint.left_lane_marking.type,
                'lane_change': waypoint.lane_change,
                'is_junction': lane.is_junction,
            }
            lanedict['left_lanes'].append(single_lane_dict)
        
        for name, lane in lanes:
            single_lane_dict = {
                'lane_id': lane.lane_id,
                'lane_type': waypoint.lane_type, 
                'lane_width': waypoint.lane_width, 
                'right_lane_color': waypoint.right_lane_marking.color, 
                'left_lane_color': waypoint.left_lane_marking.color,
                'right_lane_marking_type': waypoint.right_lane_marking.type, 
                'left_lane_marking_type': waypoint.left_lane_marking.type,
                'lane_change': waypoint.lane_change,
                'is_junction': lane.is_junction,
            }
            lanedict['ego_lane'] = single_lane_dict

        lanedict['right_lanes'] = []
        
        for name, lane in right_lanes:        
            single_lane_dict = {
                'lane_id': lane.lane_id,
                'lane_type': waypoint.lane_type, 
                'lane_width': waypoint.lane_width, 
                'right_lane_color': waypoint.right_lane_marking.color, 
                'left_lane_color': waypoint.left_lane_marking.color,
                'right_lane_marking_type': waypoint.right_lane_marking.type, 
                'left_lane_marking_type': waypoint.left_lane_marking.type,
                'lane_change': waypoint.lane_change,
                'is_junction': lane.is_junction,
            }
            lanedict['right_lanes'].append(single_lane_dict)

        lanedict['road_id'] = waypoint.road_id
        egodict = get_vehicle_attributes(self.ego, waypoint)
        
        #export data from surrounding vehicles
        if len(vehicles) > 1:
            for vehicle in vehicles:
                # TODO: change the 100m condition to field of view. 
                if vehicle.id != self.ego.id and distance(vehicle.get_location()) < 100:
                    vehicle_wp = map1.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
                    actordict[vehicle.id] = get_vehicle_attributes(vehicle, vehicle_wp)
    
        for p in pedestrians:
            if p.get_location().distance(self.ego.get_location())<100:
                ped_wp = map1.get_waypoint(p.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
                peddict[p.id]=get_actor_attributes(p, ped_wp)

        for t_light in trafficlights:
            if t_light.get_location().distance(self.ego.get_location())<100:
                lightdict[t_light.id]=get_actor_attributes(t_light)

        for s in signs:
            if s.get_location().distance(self.ego.get_location())<100:
                signdict[s.id]=get_actor_attributes(s)

        self.framedict[frame]={"ego": egodict,"actors": actordict,"pedestrians": peddict,"trafficlights": lightdict,"signs": signdict,"lane": lanedict}

        self.export_data()
        
    def export_data(self):
        if len(self.framedict)==2:
            with open(self.output_dir / (str(list(self.framedict.keys())[0]) + '-' + str(list(self.framedict.keys())[len(self.framedict)-1])+'.txt'), 'w') as file:
                file.write(json.dumps(self.framedict))
            self.framedict.clear()
