
import os
import math
import json
from collections import defaultdict
import carla

from sensors import get_actor_attributes

class DataExtractor(object):

    def __init__(self, ego):
        
        # self.world = world # This is carla world. 

        # self.output_root_dir = output_root_dir
        # self.output_dir = output_dir

        self.output_root_dir = "_out/"
        self.output_dir = "_out/data/"

        if not os.path.exists(self.output_root_dir):
            os.mkdir(self.output_root_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.framedict=defaultdict()
        self.ego = ego

    def extract_frame(self, world, frame):
        # utilities
        t = self.ego.get_transform()

        # velocity = lambda l: (3.6 * math.sqrt(l.x**2 + l.y**2 + l.z**2))
        # dv = lambda l: (3.6 * math.sqrt((l.x-v.x)**2 + (l.y-v.y)**2 + (l.z-v.z)**2))
        distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)

        vehicles = world.get_actors().filter('vehicle.*')
        pedestrians=world.get_actors().filter('walker.*')
        trafficlights=world.get_actors().filter('traffic.traffic_light')
        signs=world.get_actors().filter('traffic.traffic_sign')
        
        egodict = defaultdict()
        actordict = defaultdict()
        peddict = defaultdict()
        lightdict = defaultdict()
        signdict = defaultdict()
        lanedict = defaultdict()

        waypoint = world.get_map().get_waypoint(self.ego.get_location(),
                                                        project_to_road=True, 
                                                        lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
        lanes = [("ego_lane", waypoint), 
                    ("left_lane", waypoint.get_left_lane()), 
                    ("right_lane", waypoint.get_right_lane()), 
                    ("next_waypoint", waypoint.next(10))]  #selecting a default 10 feet ahead for the next waypoint
        for name, lane in lanes: 
            l_3d = waypoint.transform.location
            r_3d = waypoint.transform.rotation
            import pdb; pdb.set_trace()
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
                'left_lane_id': lane.get_left_lane.lane_id,
                'right_lane_id': lane.get_right_lane.lane_id,
                'is_junction': lane.is_junction,
            }
            lanedict[name] = single_lane_dict
        
        egodict = get_actor_attributes(self.ego)
        
        #export data from surrounding vehicles
        if len(vehicles) > 1:
            for vehicle in vehicles:
                # TODO: change the 100m condition to field of view. 
                if vehicle.id != self.ego.id and distance(vehicle.get_location()) < 100:
                    actordict[vehicle.id] = get_actor_attributes(vehicle)
    
        for p in pedestrians:
            if p.get_location().distance(self.ego.get_location())<100:
                peddict[p.id]=get_actor_attributes(p)

        for t_light in trafficlights:
            if t_light.get_location().distance(self.ego.get_location())<100:
                lightdict[t_light.id]=get_actor_attributes(t_light)

        for s in signs:
            if s.get_location().distance(self.ego.get_location())<100:
                signdict[s.id]=get_actor_attributes(s)

        self.framedict[frame]={"ego": egodict,"actors": actordict,"pedestrians": peddict,"trafficlights": lightdict,"signs": signdict,"lane": lanedict}

        self.export_data()
        
    def export_data(self):
        if len(self.framedict)==50:
            with open(self.output_dir + str(list(self.framedict.keys())[0]) + '-' + str(list(self.framedict.keys())[len(self.framedict)-1])+'.txt', 'w') as file:
                file.write(json.dumps(self.framedict))
            self.framedict.clear()
        