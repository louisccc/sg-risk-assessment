#REFERENCE:
#dict for lanes
# single_lane_dict = {
    # 'lane_id': lane.lane_id
    # 'location': int(l_3d.x), int(l_3d.y), int(l_3d.z)
    # 'rotation': int(r_3d.yaw), int(r_3d.roll), int(r_3d.pitch)
    # 'lane_type': waypoint.lane_type, 
    # 'lane_width': waypoint.lane_width, 
    # 'right_lane_color': waypoint.right_lane_marking.color, 
    # 'left_lane_color': waypoint.left_lane_marking.color,
    # 'right_lane_marking_type': waypoint.right_lane_marking.type, 
    # 'left_lane_marking_type': waypoint.left_lane_marking.type,
    # 'lane_change': waypoint.lane_change
    # 'left_lane_id': lane.get_left_lane.lane_id
    # 'right_lane_id': lane.get_right_lane.lane_id
    # 'is_junction': lane.is_junction
# }


#dict for all other entities
# return_dict['velocity_abs'] = int(velocity(v_3d))
# return_dict['velocity'] = int(v_3d.x), int(v_3d.y), int(v_3d.z)
# return_dict['location'] = int(l_3d.x), int(l_3d.y), int(l_3d.z)
# return_dict['rotation'] =  int(r_3d.yaw), int(r_3d.roll), int(r_3d.pitch)
# return_dict['ang_velocity'] = int(a_3d.x), int(a_3d.y), int(a_3d.z)
# return_dict['name'] = get_actor_display_name(actor)
# return_dict['lane_id'] = waypoint.lane_id     #only for moving actors
# return_dict['road_id'] = waypoint.road_id     #only for moving actors


from enum import Enum
from collections import defaultdict
import pdb
import math
import json

MOTO_NAMES = ["Harley-Davidson", "Kawasaki", "Yamaha"]
BICYCLE_NAMES = ["Gazelle", "Diamondback", "Bh"]
CAR_NAMES = ["Ford", "Bmw", "Toyota", "Nissan", "Mini", "Tesla", "Seat", "Lincoln", "Audi", "Carlamotors", "Citroen", "Mercedes-Benz", "Chevrolet", "Volkswagen", "Jeep", "Nissan", "Dodge", "Mustang"]

CAR_PROXIMITY_THRESH = 50 # max number of feet between a car and another entity to build proximity relation
MOTO_PROXIMITY_THRESH = 50
BICYCLE_PROXIMITY_THRESH = 50
PED_PROXIMITY_THRESH = 50

#defines all types of actors which can exist
#order of enum values is important as this determines which function is called. DO NOT CHANGE ENUM ORDER
class ActorType(Enum):
    CAR = 0 #26, 142, 137:truck
    MOTO = 1 #80
    BICYCLE = 2 #11
    PED = 3 #90, 91, 98: "player", 78:man, 79:men, 149:woman, 56: guy, 53: girl
    LANE = 4 #124:street, 114:sidewalk
    LIGHT = 5 # 99: "pole", 76: light
    SIGN = 6
    ROAD = 7
    
ACTOR_NAMES=['car','moto','bicycle','ped','lane','light','sign', 'road']
    
class Relations(Enum):
    isIn = 0
    near = 1
    partOf = 2
    instanceOf = 3
    hasAttribute = 4
    frontLeft = 5
    frontRight = 6
    rearLeft = 7
    rearRight = 8

#This class extracts relations for every pair of entities in a scene
class RelationExtractor:

    def get_actor_type(self, actor):
        if "lane_type" in actor.attr.keys():
            return ActorType.LANE
        if actor.attr["name"] == "Traffic Light":
            return ActorType.LIGHT
        if actor.attr["name"].split(" ")[0] == "Pedestrian":
            return ActorType.PED
        if actor.attr["name"].split(" ")[0] in CAR_NAMES:
            return ActorType.CAR
        if actor.attr["name"].split(" ")[0] in MOTO_NAMES:
            return ActorType.MOTO
        if actor.attr["name"].split(" ")[0] in BICYCLE_NAMES:
            return ActorType.BICYCLE
        if "Sign" in actor.attr["name"]:
            return ActorType.SIGN
            
        print(actor.attr)
        import pdb; pdb.set_trace()
        raise NameError("Actor name not found for actor with name: " + actor.attr["name"])
            
    #takes in two entities and extracts all relations between those two entities. extracted relations are bidirectional    
    def extract_relations(self, actor1, actor2):
        #import pdb; pdb.set_trace()
        type1 = self.get_actor_type(actor1)
        type2 = self.get_actor_type(actor2)
        
        low_type = min(type1.value, type2.value) #the lower of the two enums.
        high_type = max(type1.value, type2.value)
    
        function_call = "self.extract_relations_"+ACTOR_NAMES[low_type]+"_"+ACTOR_NAMES[high_type]+"(actor1, actor2) if type1.value <= type2.value "\
                        "else self.extract_relations_"+ACTOR_NAMES[low_type]+"_"+ACTOR_NAMES[high_type]+"(actor2, actor1)"
        return eval(function_call)
           

#~~~~~~~~~specific relations for each pair of actors possible~~~~~~~~~~~~
#actor 1 corresponds to the first actor in the function name and actor2 the second

    def extract_relations_car_car(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])

        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
            
    def extract_relations_car_lane(self, actor1, actor2):
        relation_list = []
        # import pdb; pdb.set_trace()
        if(self.in_lane(actor1,actor2)):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_car_light(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_car_sign(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_car_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_car_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_car_moto(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
        
    def extract_relations_moto_moto(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_moto_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_moto_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_moto_lane(self, actor1, actor2):
        relation_list = []
        if(self.in_lane(actor1,actor2)):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_moto_light(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_moto_sign(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        

    def extract_relations_bicycle_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < BICYCLE_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < BICYCLE_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_lane(self, actor1, actor2):
        relation_list = []
        if(self.in_lane(actor1,actor2)):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_bicycle_light(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_sign(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
        
    def extract_relations_ped_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < PED_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
           
    def extract_relations_ped_lane(self, actor1, actor2):
        relation_list = []
        if(self.in_lane(actor1,actor2)):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_ped_light(self, actor1, actor2):
        relation_list = []
        #proximity relation could indicate ped waiting for crosswalk at a light
        if(self.euclidean_distance(actor1, actor2) < PED_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_ped_sign(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        

    def extract_relations_lane_lane(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_lane_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_lane_sign(self, actor1, actor2):
        relation_list = []
        return relation_list

    def extract_relations_light_light(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_light_sign(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list

    def extract_relations_sign_sign(self, actor1, actor2):
        relation_list = []
        relation_list.append(self.extract_directional_relation(actor1, actor2))
        relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
    
    
#~~~~~~~~~~~~~~~~~~UTILITY FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~
    #return euclidean distance between actors
    def euclidean_distance(self, actor1, actor2):
        #import pdb; pdb.set_trace()
        l1 = actor1.attr['location']
        l2 = actor2.attr['location']
        return math.sqrt((l1[0] - l2[0])**2 + (l1[1]- l2[1])**2 + (l1[2] - l2[2])**2)
        
    #check if an actor is in a certain lane
    def in_lane(self, actor1, actor2):
        if('lane_id' in actor1.attr.keys() and actor1.attr['lane_id'] == actor2.attr['lane_id']):
            return True
        else:
            return False
    
    def extract_directional_relation(self, actor1, actor2):
        x1 = actor1.attr['location'][0]
        x2 = actor2.attr['location'][0]
        y1 = actor1.attr['location'][1]
        y2 = actor2.attr['location'][1]
        
        x_diff = x2 - x1
        y_diff = y2 - y1
        if (x_diff < 0):
            if (y_diff < 0):
                return [actor1, Relations.rearLeft, actor2]
            else:
                return [actor1, Relations.rearRight, actor2]
        else:
            if (y_diff < 0):
                return [actor1, Relations.frontLeft, actor2]
            else:
                return [actor1, Relations.frontRight, actor2]