import sys, random, math, json, carla, py_trees
import sensors
from sensors import get_actor_attributes, get_vehicle_attributes
from pathlib import Path
from collections import defaultdict

SRUNNER_PATH = r'./scenario_runner'
sys.path.append(SRUNNER_PATH)
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import LaneChange


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
        self.new_path = None
        
        self.num_of_existing_datapoints = len(list(self.root_path.glob('*')))
        self.dir_index = 0

        # This indicates the state of ego's lane change behavior
        self.lane_changing= False

        self.client = client

        self.weather_presets = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon, 
                carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetCloudyNoon, 
                carla.WeatherParameters.SoftRainNoon, carla.WeatherParameters.MidRainyNoon, 
                carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.ClearSunset, 
                carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetSunset, 
                carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.SoftRainSunset, 
                carla.WeatherParameters.MidRainSunset, carla.WeatherParameters.HardRainSunset]
       
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

        self.sensors_dict["camera_manager"] = sensors.CameraManager(self.ego, gamma, dimensions, root_path)
        self.sensors_dict["camera_manager"].transform_index = cam_pos_index
        self.sensors_dict["camera_manager"].set_sensor(cam_index, notify=False)
        self.sensors_dict["lane_invasion"] = sensors.LaneInvasionDetector(self.ego, root_path)
        # self.sensors_dict["camera_manager_ss"] = sensors.CameraManager(self.ego, gamma, dimensions, root_path)
        # self.sensors_dict["camera_manager_ss"].transform_index = cam_pos_index
        # self.sensors_dict["camera_manager_ss"].set_sensor(cam_index+5, notify=False)

    def destroy_sensors(self):
        for _, sensor in self.sensors_dict.items():
            sensor.destroy()
        self.sensors_dict = {}
        
    def toggle_recording(self):
        for _, sensor in self.sensors_dict.items():
            sensor.toggle_recording()

    def tick(self, frame_num):
        self.tick_count += 1
        
        if self.tick_count == 100:
            # set random weather
            self.carla_world.set_weather(random.choice(self.weather_presets))

            # choose random vehicle and prepare for recording
            print("Picking vehicle and attaching sensors...")
            self.ego = self.carla_world.get_actor(random.choice(self.vehicles_list))
            self.carla_world.get_spectator().set_transform(self.ego.get_transform())

            print("Attempting lane change...")
            self.lane_change_direction = None
            
            # check available lane changes
            waypoint = self.map.get_waypoint(self.ego.get_location())
            velocity = self.ego.get_velocity()

            if ( waypoint.lane_change == carla.LaneChange.NONE or 
                (abs(velocity.x) <= 1.0 and abs(velocity.y) <= 1.0) ):
                print("Lane Change not available.")
                self.tick_count = 0
                return
            elif (waypoint.lane_change == carla.LaneChange.Both):
                print("Both")
                self.lane_change_direction = random.choice(['left', 'right'])
            elif (waypoint.lane_change == carla.LaneChange.Left):
                print("Left")
                self.lane_change_direction = 'left'
            else:
                print("Right")
                self.lane_change_direction = 'right'

            print("Start Lane Changing and Recording...")
            self.lane_changing= True 

            self.new_path = "%s/%s" % (str(self.root_path), self.num_of_existing_datapoints + self.dir_index)
            self.extractor = DataExtractor(self.ego, self.new_path)
            self.attach_sensors(self.new_path)
            self.dir_index += 1
            self.toggle_recording()

            ## setting 
            self.lane_change_controller = LaneChange(self.ego, direction=self.lane_change_direction)
            self.lane_change_controller.initialise()

            self.client.apply_batch_sync([carla.command.SetAutopilot(self.ego, False)], True)
              
        if self.lane_changing:
            ####### collect the lane invasion data. 
            ## pull the result from self.sensors_dict["lane_invasion"] 
            # and update to self.extracotr using API?
            self.extractor.extract_frame(self.carla_world, self.map, frame_num)
            success = self.lane_change_controller.update()
            if success == py_trees.common.Status.SUCCESS or self.tick_count > 350:
                #write to metadata file
                with open((Path(self.new_path) / 'metadata.txt').resolve(),'w') as file:
                    weather=self.carla_world.get_weather()
                    
                    metadata_dict={"wetness":weather.wetness,"wind_intensity":weather.wind_intensity,"precipitation_deposits":weather.precipitation_deposits,
                    "precipitation": weather.precipitation,"cloudiness": weather.cloudiness,"fog_density": weather.fog_density,"fog_distance": weather.fog_distance,
                    "sun_altitude_angle": weather.sun_altitude_angle,"sun_azimuth_angle": weather.sun_azimuth_angle}
                    
                    file.write(json.dumps(metadata_dict))


                self.lane_changing = False
                print('set set_autopilot back to true')
                self.client.apply_batch_sync([carla.command.SetAutopilot(self.ego, True)], True)
                self.toggle_recording()
                print("Cleaning up sensors...")
                self.destroy_sensors()
                self.tick_count = 0

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

        ego_lane = waypoint
        left_lanes = []
        right_lanes = [] 
        
        left_lane = waypoint 
        while True: 
            lane = left_lane.get_left_lane()
            if lane is None:
                break 
            left_lanes.append(("lane", lane))
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
                    
        lanedict['left_lanes'] = []
        for name, lane in left_lanes:
            single_lane_dict = {
                'lane_id': lane.lane_id,
                'lane_type': lane.lane_type, 
                'lane_width': lane.lane_width, 
                'right_lane_color': lane.right_lane_marking.color, 
                'left_lane_color': lane.left_lane_marking.color,
                'right_lane_marking_type': lane.right_lane_marking.type, 
                'left_lane_marking_type': lane.left_lane_marking.type,
                'lane_change': lane.lane_change,
                'is_junction': lane.is_junction,
            }
            lanedict['left_lanes'].append(single_lane_dict)
        
        lanedict['ego_lane'] = {
            'lane_id': ego_lane.lane_id,
            'lane_type': ego_lane.lane_type, 
            'lane_width': ego_lane.lane_width, 
            'right_lane_color': ego_lane.right_lane_marking.color, 
            'left_lane_color': ego_lane.left_lane_marking.color,
            'right_lane_marking_type': ego_lane.right_lane_marking.type, 
            'left_lane_marking_type': ego_lane.left_lane_marking.type,
            'lane_change': ego_lane.lane_change,
            'is_junction': ego_lane.is_junction,
        }

        lanedict['right_lanes'] = []
        
        for name, lane in right_lanes:        
            single_lane_dict = {
                'lane_id': lane.lane_id,
                'lane_type': lane.lane_type, 
                'lane_width': lane.lane_width, 
                'right_lane_color': lane.right_lane_marking.color, 
                'left_lane_color': lane.left_lane_marking.color,
                'right_lane_marking_type': lane.right_lane_marking.type, 
                'left_lane_marking_type': lane.left_lane_marking.type,
                'lane_change': lane.lane_change,
                'is_junction': lane.is_junction,
            }
            lanedict['right_lanes'].append(single_lane_dict)

        lanedict['road_id'] = waypoint.road_id
        egodict = get_vehicle_attributes(self.ego, waypoint)
        
        # export data from surrounding vehicles
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
