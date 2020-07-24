from enum import Enum
import math


MOTO_NAMES = ["Harley-Davidson", "Kawasaki", "Yamaha"]
BICYCLE_NAMES = ["Gazelle", "Diamondback", "Bh"]
CAR_NAMES = ["Ford", "Bmw", "Toyota", "Nissan", "Mini", "Tesla", "Seat", "Lincoln", "Audi", "Carlamotors", "Citroen", "Mercedes-Benz", "Chevrolet", "Volkswagen", "Jeep", "Nissan", "Dodge", "Mustang"]

CAR_PROXIMITY_THRESH_NEAR_COLL = 4
CAR_PROXIMITY_THRESH_SUPER_NEAR = 7 # max number of feet between a car and another entity to build proximity relation
CAR_PROXIMITY_THRESH_VERY_NEAR = 10
CAR_PROXIMITY_THRESH_NEAR = 16
CAR_PROXIMITY_THRESH_VISIBLE = 25
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
    near_coll = 1
    super_near = 2
    very_near = 3
    near = 4
    visible = 5
    inDFrontOf = 6
    inSFrontOf = 7
    atDRearOf = 8
    atSRearOf = 9
    toLeftOf = 10
    toRightOf = 11

RELATION_COLORS = ["black", "red", "orange", "yellow", "green", "purple", "blue", 
                "sienna", "pink", "pink", "pink",  "turquoise", "turquoise", "turquoise", "violet", "violet"]

#This class extracts relations for every pair of entities in a scene
class RelationExtractor:
    def __init__(self, ego_node):
        self.ego_node = ego_node 

    def get_actor_type(self, actor):
        if "curr" in actor.attr.keys():
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

        # import pdb; pdb.set_trace()
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
        # consider the proximity relations with neighboring lanes.
        if actor1.name.startswith("ego:") or actor2.name.startswith("ego:"):
            if self.euclidean_distance(actor1, actor2) <= CAR_PROXIMITY_THRESH_NEAR:
                relation_list += self.create_proximity_relations(actor1, actor2)
                relation_list += self.create_proximity_relations(actor2, actor1)
                relation_list += self.extract_directional_relation(actor1, actor2)
                relation_list += self.extract_directional_relation(actor2, actor1)
        return relation_list
            
    def extract_relations_car_lane(self, actor1, actor2):
        relation_list = []
        # if(self.in_lane(actor1,actor2)):
        #     relation_list.append([actor1, Relations.isIn, actor2])
            
        return relation_list 
        
    def extract_relations_car_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_sign(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_ped(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_bicycle(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_moto(self, actor1, actor2):
        relation_list = []
        return relation_list
        
        
    def extract_relations_moto_moto(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_bicycle(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_ped(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_lane(self, actor1, actor2):
        relation_list = []
        # if(self.in_lane(actor1,actor2)):
        #     relation_list.append([actor1, Relations.isIn, actor2])
        #     # relation_list.append([actor2, Relations.isIn, actor1])
        return relation_list 
        
    def extract_relations_moto_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_sign(self, actor1, actor2):
        relation_list = []
        return relation_list
        

    def extract_relations_bicycle_bicycle(self, actor1, actor2):
        relation_list = []
        # if(self.euclidean_distance(actor1, actor2) < BICYCLE_PROXIMITY_THRESH):
        #     relation_list.append([actor1, Relations.near, actor2])
        #     relation_list.append([actor2, Relations.near, actor1])
        #     #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #     #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_ped(self, actor1, actor2):
        relation_list = []
        # if(self.euclidean_distance(actor1, actor2) < BICYCLE_PROXIMITY_THRESH):
        #     relation_list.append([actor1, Relations.near, actor2])
        #     relation_list.append([actor2, Relations.near, actor1])
        #     #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #     #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_lane(self, actor1, actor2):
        relation_list = []
        # if(self.in_lane(actor1,actor2)):
        #     relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_bicycle_light(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_sign(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_ped_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < PED_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
            relation_list.append([actor2, Relations.near, actor1])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
           
    def extract_relations_ped_lane(self, actor1, actor2):
        relation_list = []
        # if(self.in_lane(actor1,actor2)):
        #     relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_ped_light(self, actor1, actor2):
        relation_list = []
        #proximity relation could indicate ped waiting for crosswalk at a light
        # if(self.euclidean_distance(actor1, actor2) < PED_PROXIMITY_THRESH):
        #     relation_list.append([actor1, Relations.near, actor2])
        #     relation_list.append([actor2, Relations.near, actor1])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_ped_sign(self, actor1, actor2):
        relation_list = []
        # relation_list.append(self.extract_directional_relation(actor1, actor2))
        # relation_list.append(self.extract_directional_relation(actor2, actor1))
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
        return relation_list
        
    def extract_relations_light_sign(self, actor1, actor2):
        relation_list = []
        return relation_list

    def extract_relations_sign_sign(self, actor1, actor2):
        relation_list = []
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
        if 'lane_idx' in actor1.attr.keys():
            # calculate the distance bewteen actor1 and actor2
            # if it is below 3.5 then they have is in relation.
                # if actor1 is ego: if actor2 is not equal to the ego_lane's index then it's invading relation.
            if actor1.attr['lane_idx'] == actor2.attr['lane_idx']:
                return True
            if "invading_lane" in actor1.attr:
                if actor1.attr['invading_lane'] == actor2.attr['lane_idx']:
                    return True
                if "orig_lane_idx" in actor1.attr:
                    if actor1.attr['orig_lane_idx'] == actor2.attr['lane_idx']:
                        return True
        else:
            return False
    
    def create_proximity_relations(self, actor1, actor2):
        if self.euclidean_distance(actor1, actor2) <= CAR_PROXIMITY_THRESH_NEAR_COLL:
            return [[actor1, Relations.near_coll, actor2]]
        elif self.euclidean_distance(actor1, actor2) <= CAR_PROXIMITY_THRESH_SUPER_NEAR:
            return [[actor1, Relations.super_near, actor2]]
        elif self.euclidean_distance(actor1, actor2) <= CAR_PROXIMITY_THRESH_VERY_NEAR:
            return [[actor1, Relations.very_near, actor2]]
        elif self.euclidean_distance(actor1, actor2) <= CAR_PROXIMITY_THRESH_NEAR:
            return [[actor1, Relations.near, actor2]]
        elif self.euclidean_distance(actor1, actor2) <= CAR_PROXIMITY_THRESH_VISIBLE:
            return [[actor1, Relations.visible, actor2]]
        return []

    def extract_directional_relation(self, actor1, actor2):
        relation_list = []
        # gives directional relations between actors based on their 2D absolute positions.      
        x1, y1 = math.cos(math.radians(actor1.attr['rotation'][0])), math.sin(math.radians(actor1.attr['rotation'][0]))
        x2, y2 = actor2.attr['location'][0] - actor1.attr['location'][0], actor2.attr['location'][1] - actor1.attr['location'][1]
        x2, y2 = x2 / math.sqrt(x2**2+y2**2), y2 / math.sqrt(x2**2+y2**2)

        degree = math.degrees(math.atan2(y1, x1)) - math.degrees(math.atan2(y2, x2)) 
        if degree < 0: 
            degree += 360
            
        if degree <= 45: # actor2 is in front of actor1
            relation_list.append([actor1, Relations.atDRearOf, actor2])
        elif degree >= 45 and degree <= 90:
            relation_list.append([actor1, Relations.atSRearOf, actor2])
        elif degree >= 90 and degree <= 135:
            relation_list.append([actor1, Relations.inSFrontOf, actor2])
        elif degree >= 135 and degree <= 180: # actor2 is behind actor1
            relation_list.append([actor1, Relations.inDFrontOf, actor2])
        elif degree >= 180 and degree <= 225: # actor2 is behind actor1
            relation_list.append([actor1, Relations.inDFrontOf, actor2])
        elif degree >= 225 and degree <= 270:
            relation_list.append([actor1, Relations.inSFrontOf, actor2])
        elif degree >= 270 and degree <= 315:
            relation_list.append([actor1, Relations.atSRearOf, actor2])
        elif degree >= 315 and degree <= 360: 
            relation_list.append([actor1, Relations.atDRearOf, actor2])

        if actor2.attr['lane_idx'] < actor1.attr['lane_idx']: # actor2 to the left of actor1 
            relation_list.append([actor1, Relations.toRightOf, actor2])
        elif actor2.attr['lane_idx'] > actor1.attr['lane_idx']: # actor2 to the right of actor1 
            relation_list.append([actor1, Relations.toLeftOf, actor2])

        return relation_list