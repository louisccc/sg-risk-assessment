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


from enum import Enum
from collections import defaultdict
import pdb

MOTO_NAMES = ["Harley-Davidson", "Kawasaki", "Yamaha"]
BICYCLE_NAMES = ["Gazelle", "Diamondback", "Bh"]
CAR_NAMES = ["Ford", "Bmw", "Toyota", "Nissan", "Mini", "Tesla", "Seat", "Lincoln", "Audi", "Carlamotors", "Citroen", "Mercedes-Benz", "Chevrolet", "Volkswagen", "Jeep", "Nissan", "Dodge"]

CAR_PROXIMITY_THRESH = 10 # max number of feet between a car and another entity to build proximity relation
MOTO_PROXIMITY_THRESH = 10
BICYCLE_PROXIMITY_THRESH = 10
PED_PROXIMITY_THRESH = 10

#defines all types of actors which can exist
#order of enum values is important as this determines which function is called. DO NOT CHANGE ENUM ORDER
class ActorType(Enum):
    CAR = 0
    MOTO = 1
    BICYCLE = 2
    PED = 3 
    LIGHT = 4
    SIGN = 5
    LANE = 6
    
class Relations(Enum):
    isIn = 0
    near = 1
    partOf = 2
    instanceOf = 3

#This class extracts relations for every pair of entities in a scene
class RelationExtractor:

    def get_actor_type(self, actor):
            
        if "lane_id" in actor.keys():
            return ActorType.LANE
        if actor["name"].split(" ")[0] in CAR_NAMES:
            return ActorType.CAR
        if actor["name"].split(" ")[0] in MOTO_NAMES:
            return ActorType.MOTO
        if actor["name"].split(" ")[0] in BICYCLE_NAMES:
            return ActorType.BICYCLE
        raise NameError("Actor name not found for actor with name: " + actor["name"])
            
    #takes in two entities and extracts all relations between those two entities. extracted relations are bidirectional    
    def extract_relations(self, actor1, actor2):
        type1 = self.get_actor_type(actor1)
        type2 = self.get_actor_type(actor2)
        
        if type1 == type2:
            if type1 == ActorType.CAR:
                return self.extract_relations_car_car(actor1, actor2)
            if type1 == ActorType.MOTO:
                return self.extract_relations_moto_moto(actor1, actor2)
            if type1 == ActorType.BICYCLE:
                return self.extract_relations_bicycle_bicycle(actor1, actor2)
            if type1 == ActorType.PED:
                return self.extract_relations_ped_ped(actor1, actor2)
        
        low_type = min(type1.value, type2.value) #the lower of the two enums.
        high_type = max(type1.value, type2.value)
        
        if low_type == ActorType.CAR.value:
            if high_type == ActorType.MOTO.value:
                return self.extract_relations_car_moto(actor1, actor2) if type1.value < type2.value else self.extract_relations_car_moto(actor2, actor1)
            if high_type == ActorType.BICYCLE:
                return self.extract_relations_car_bicycle(actor1, actor2) if type1.value < type2.value else self.extract_relations_car_bicycle(actor2, actor1)
            if high_type == ActorType.PED:
                return self.extract_relations_car_ped(actor1, actor2) if type1.value < type2.value else self.extract_relations_car_ped(actor2, actor1)
            if high_type == ActorType.LANE:
                return self.extract_relations_car_lane(actor1, actor2) if type1.value < type2.value else self.extract_relations_car_lane(actor2, actor1)
            if high_type == ActorType.LIGHT:
                return self.extract_relations_car_light(actor1, actor2) if type1.value < type2.value else self.extract_relations_car_lane(actor2, actor1)
            if high_type == ActorType.SIGN:
                return self.extract_relations_car_sign(actor1, actor2) if type1.value < type2.value else self.extract_relations_car_sign(actor2, actor1)
            
        if low_type == ActorType.MOTO:
            if high_type == ActorType.BICYCLE:
                return self.extract_relations_moto_bicycle(actor1, actor2) if type1.value < type2.value else self.extract_relations_moto_bicycle(actor2, actor1)
            if high_type == ActorType.PED:
                return self.extract_relations_moto_ped(actor1, actor2) if type1.value < type2.value else self.extract_relations_moto_ped(actor2, actor1)
            if high_type == ActorType.LANE:
                return self.extract_relations_moto_lane(actor1, actor2) if type1.value < type2.value else self.extract_relations_moto_lane(actor2, actor1)
            if high_type == ActorType.LIGHT:
                return self.extract_relations_moto_light(actor1, actor2) if type1.value < type2.value else self.extract_relations_moto_light(actor2, actor1)
            if high_type == ActorType.SIGN:
                return self.extract_relations_moto_sign(actor1, actor2) if type1.value < type2.value else self.extract_relations_moto_sign(actor2, actor1)
            
        if low_type == ActorType.BICYCLE:
            if high_type == ActorType.PED:
                return self.extract_relations_bicycle_ped(actor1, actor2) if type1.value < type2.value else self.extract_relations_bicycle_ped(actor2, actor1)
            if high_type == ActorType.LANE:
                return self.extract_relations_bicycle_lane(actor1, actor2) if type1.value < type2.value else self.extract_relations_bicycle_lane(actor2, actor1)
            if high_type == ActorType.LIGHT:
                return self.extract_relations_bicycle_light(actor1, actor2) if type1.value < type2.value else self.extract_relations_bicycle_light(actor2, actor1)
            if high_type == ActorType.SIGN:
                return self.extract_relations_bicycle_sign(actor1, actor2) if type1.value < type2.value else self.extract_relations_bicycle_sign(actor2, actor1)
                
        if low_type == ActorType.PED:
            if high_type == ActorType.LANE:
                return self.extract_relations_ped_lane(actor1, actor2) if type1.value < type2.value else self.extract_relations_ped_lane(actor2, actor1)
            if high_type == ActorType.LIGHT:
                return self.extract_relations_ped_light(actor1, actor2) if type1.value < type2.value else self.extract_relations_ped_light(actor2, actor1)            
            if high_type == ActorType.SIGN:
                return self.extract_relations_ped_sign(actor1, actor2) if type1.value < type2.value else self.extract_relations_ped_sign(actor2, actor1)
                
        if low_type == ActorType.LANE:
            if high_type == ActorType.LIGHT:
                return self.extract_relations_lane_light(actor1, actor2) if type1.value < type2.value else self.extract_relations_lane_light(actor2, actor1)
            if high_type == ActorType.SIGN:
                return self.extract_relations_lane_sign(actor1, actor2) if type1.value < type2.value else self.extract_relations_lane_sign(actor2, actor1)                

#~~~~~~~~~specific relations for each pair of actors possible~~~~~~~~~~~~
#actor 1 corresponds to the first actor in the function name and actor2 the second

    def extract_relations_car_car(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
            
    def extract_relations_car_lane(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < actor2['lane_width']/2):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_car_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_sign(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
    def extract_relations_car_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
    def extract_relations_car_moto(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
        
    def extract_relations_moto_moto(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
    def extract_relations_moto_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
    def extract_relations_moto_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
    def extract_relations_moto_lane(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < actor2['lane_width']/2):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_moto_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_sign(self, actor1, actor2):
        relation_list = []
        return relation_list
        

    def extract_relations_bicycle_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < BICYCLE_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
    def extract_relations_bicycle_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
    def extract_relations_bicycle_lane(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < actor2['lane_width']/2):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_bicycle_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_bicycle_sign(self, actor1, actor2):
        relation_list = []
        return relation_list
        
        
    def extract_relations_ped_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < PED_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
           
    def extract_relations_ped_lane(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < actor2['lane_width']/2):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_ped_light(self, actor1, actor2):
        relation_list = []
        #proximity relation could indicate ped waiting for crosswalk at a light
        if(self.euclidean_distance(actor1, actor2) < PED_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
        return relation_list
        
    def extract_relations_ped_sign(self, actor1, actor2):
        relation_list = []
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

    
#~~~~~~~~~~~~~~~~~~UTILITY FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~
    #return euclidean distance between actors
    def euclidean_distance(self, actor1, actor2):
        l1 = actor1['location']
        l2 = actor2['location']
        return sqrt((l1[0] - l2[0])**2 + (l1[1]- l2[1])**2 + (l1[2] - l2[2])**2)
    
    
    
if __name__ == "__main__":
    #demo code
    d1 = defaultdict()
    d2 = defaultdict()
    d1['name'] = "Ford Mustang"
    d2['name'] = "Kawasaki Ninja"
    r = RelationExtractor()
    l = r.extract_relations(d1,d2)
    print(l)