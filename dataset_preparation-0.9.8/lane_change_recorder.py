import random
import sensors
from pathlib import Path
class LaneChangeRecorder:

    def __init__(self, traffic_manager, carla_world):
        # It has Idle, 
        self.state = "Idle"
        self.tick_count = 0
        self.carla_world = carla_world
        self.vehicles_list = []
        self.traffic_manager = traffic_manager
        self.sensors_dict = {}
        self.root_path = Path("./_out")

        self.num_of_existing_datapoints = len(list(self.root_path.glob('*')))
        self.dir_index = 0
       
    def set_vehicles_list(self, vehicles_list):
        self.vehicles_list = vehicles_list

    def attach_sensors(self, root_path):
        """
        Spawn and attach sensors to ego vehicles
        """
        cam_index = 0
        cam_pos_index = 1
        dimensions = [1280, 720]
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

    def tick(self):
        self.tick_count += 1

        if self.tick_count == 25:
            # stop recording and clean up sensors
            self.toggle_recording()
            print("Cleaning up sensors...")
            for _, sensor in self.sensors_dict.items():
                sensor.destroy()
            self.sensors_dict = {}

        elif self.tick_count == 40:
            # choose random vehicle and prepare for recording
            print("Attach sensors and start recording...")
            self.ego = self.carla_world.get_actor(random.choice(self.vehicles_list))
            new_path = "%s/%s" % (str(self.root_path), self.num_of_existing_datapoints + self.dir_index)
            self.attach_sensors(new_path)
            self.dir_index += 1

        elif self.tick_count >= 45:
            print("Changing Lane...")
            self.toggle_recording()
            self.traffic_manager.force_lane_change(self.ego, random.choice([True, False]))
            self.tick_count = 0