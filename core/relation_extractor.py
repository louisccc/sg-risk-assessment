from enum import Enum
import math


MOTO_NAMES = ["Harley-Davidson", "Kawasaki", "Yamaha"]
BICYCLE_NAMES = ["Gazelle", "Diamondback", "Bh"]
CAR_NAMES = ["Ford", "Bmw", "Toyota", "Nissan", "Mini", "Tesla", "Seat", "Lincoln", "Audi", "Carlamotors", "Citroen", "Mercedes-Benz", "Chevrolet", "Volkswagen", "Jeep", "Nissan", "Dodge", "Mustang"]

CAR_PROXIMITY_THRESH_SUPER_NEAR = 10 # max number of feet between a car and another entity to build proximity relation
CAR_PROXIMITY_THRESH_VERY_NEAR = 25
CAR_PROXIMITY_THRESH_NEAR = 50
CAR_PROXIMITY_THRESH_VISIBLE = 100
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
    super_near = 1
    very_near = 2
    near = 3
    visible = 4
    partOf = 5
    instanceOf = 6
    hasAttribute = 7
    rear = 8
    front = 9
    frontLeft = 10
    frontRight = 11
    rearLeft = 12
    rearRight = 13

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
        if actor1.name.startswith("ego:") or actor2.name.startswith("ego:"):
            if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_SUPER_NEAR):
                relation_list.append([actor1, Relations.super_near, actor2])
            elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VERY_NEAR):
                relation_list.append([actor1, Relations.very_near, actor2])
            elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_NEAR):
                relation_list.append([actor1, Relations.near, actor2])
            elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VISIBLE):
                relation_list.append([actor1, Relations.visible, actor2])

            # import pdb; pdb.set_trace()

            if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_NEAR):
                relation_list.append(self.extract_directional_relation(actor1, actor2))
                relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
            
    def extract_relations_car_lane(self, actor1, actor2):
        relation_list = []
        # import pdb; pdb.set_trace()
        if(self.in_lane(actor1,actor2)):
            relation_list.append([actor1, Relations.isIn, actor2])
            # import pdb; pdb.set_trace()
        return relation_list 
        
    def extract_relations_car_light(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_car_sign(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_car_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_SUPER_NEAR):
            relation_list.append([actor1, Relations.super_near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VERY_NEAR):
            relation_list.append([actor1, Relations.very_near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_NEAR):
            relation_list.append([actor1, Relations.near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VISIBLE):
            relation_list.append([actor1, Relations.visible, actor2])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_car_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_SUPER_NEAR):
            relation_list.append([actor1, Relations.super_near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VERY_NEAR):
            relation_list.append([actor1, Relations.very_near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_NEAR):
            relation_list.append([actor1, Relations.near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VISIBLE):
            relation_list.append([actor1, Relations.visible, actor2])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_car_moto(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_SUPER_NEAR):
            relation_list.append([actor1, Relations.super_near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VERY_NEAR):
            relation_list.append([actor1, Relations.very_near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_NEAR):
            relation_list.append([actor1, Relations.near, actor2])
        elif(self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VISIBLE):
            relation_list.append([actor1, Relations.visible, actor2])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
        
    def extract_relations_moto_moto(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_moto_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_moto_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < MOTO_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_moto_lane(self, actor1, actor2):
        relation_list = []
        if(self.in_lane(actor1,actor2)):
            relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_moto_light(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_moto_sign(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        

    def extract_relations_bicycle_bicycle(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < BICYCLE_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < BICYCLE_PROXIMITY_THRESH):
            relation_list.append([actor1, Relations.near, actor2])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_lane(self, actor1, actor2):
        relation_list = []
        if(self.in_lane(actor1,actor2)):
            relation_list.append([actor1, Relations.isIn, actor2])
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
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
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
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_ped_sign(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
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
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_light_sign(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list

    def extract_relations_sign_sign(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
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
            # import pdb; pdb.set_trace()
            if actor1.attr['lane_idx'] == actor2.attr['lane_idx']:
                return True
            if "invading_lane" in actor1.attr:
                if actor1.attr['invading_lane'] == actor2.attr['lane_idx']:
                    return True
        else:
            return False
    
    #gives directional relations between actors based on their 2D absolute positions.
    #TODO: fix these relations, since the locations are based on the world coordinate system and are not relative to ego.
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

        pass