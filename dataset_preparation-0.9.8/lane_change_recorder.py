import random
import sensors
from pathlib import Path

import os, math, json
from collections import defaultdict
import carla

from sensors import get_actor_attributes, get_vehicle_attributes

class LaneChangeRecorder:

    def __init__(self, traffic_manager, carla_world):
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
        
    def toggle_recording(self):

        if 'camera_manager' in self.sensors_dict:
            self.sensors_dict["camera_manager"].toggle_recording()

        if 'camera_manager_ss' in self.sensors_dict:
            self.sensors_dict["camera_manager_ss"].toggle_recording()

    def tick(self, frame_num):
        self.tick_count += 1

        if self.tick_count == 50:
            # choose random vehicle and prepare for recording
            print("Attach sensors and start recording...")
            self.ego = self.carla_world.get_actor(random.choice(self.vehicles_list))
            new_path = "%s/%s" % (str(self.root_path), self.num_of_existing_datapoints + self.dir_index)
            self.extractor = DataExtractor(self.ego, new_path)
            self.attach_sensors(new_path)
            self.dir_index += 1

        elif self.tick_count == 100:
            print("Changing Lane...")
            self.toggle_recording()
            self.is_recording = True
            print(self.ego.get_velocity())
            self.traffic_manager.force_lane_change(self.ego, random.choice([True, False]))
        
        elif self.tick_count >= 200:
            # stop recording and clean up sensors
            self.toggle_recording()
            print("Cleaning up sensors...")
            for _, sensor in self.sensors_dict.items():
                sensor.destroy()
            self.sensors_dict = {}
            self.tick_count = 0
            self.is_recording = False


        if self.is_recording:
            self.extractor.extract_frame(self.carla_world, self.map, frame_num)

class DataExtractor(object):

    def __init__(self, ego, store_path):
        
        # self.world = world # This is carla world. 

        # self.output_root_dir = output_root_dir
        # self.output_dir = output_dir

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
                 # ("left_lane", waypoint.get_left_lane()), 
                 # ("right_lane", waypoint.get_right_lane()), 
                 # ("next_waypoint", waypoint.next(10))
                 ]  #selecting a default 10 feet ahead for the next waypoint

        for name, lane in lanes:
            # print(name, lane) 
            l_3d = waypoint.transform.location
            r_3d = waypoint.transform.rotation
            
            single_lane_dict = {
                'lane_id': lane.lane_id,
                'location': [int(l_3d.x), int(l_3d.y), int(l_3d.z)],
                'rotation': [int(r_3d.yaw), int(r_3d.roll), int(r_3d.pitch)],
                'lane_type': waypoint.lane_type, 
                'lane_width': waypoint.lane_width, 
                'right_lane_color': waypoint.right_lane_marking.color, 
                'left_lane_color': waypoint.left_lane_marking.color,
                'right_lane_marking_type': waypoint.right_lane_marking.type, 
                'left_lane_marking_type': waypoint.left_lane_marking.type,
                'lane_change': waypoint.lane_change,
                # 'left_lane_id': lane.get_left_lane().lane_id,
                # 'right_lane_id': lane.get_right_lane().lane_id,
                'is_junction': lane.is_junction,
            }
            lanedict[name] = single_lane_dict
        
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
        