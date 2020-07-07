# -*- coding: utf-8 -*-
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import sys, os, pdb
import numpy as np
import cv2
import random
import networkx as nx
import itertools
import math
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# panoptic seg
sys.path.append(os.path.dirname(sys.path[0]))
from core.relation_extractor import ActorType
from core.lane_extractor import LaneExtractor

coco_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
'bus', 'train', 'truck', 'boat', 'traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
'kite', 'baseball bat', 'baseball glove', 'skateboard',
'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
'teddy bear', 'hair drier', 'toothbrush']


from enum import Enum

#SETTINGS FOR 640x360 CARLA IMAGES:
# CARLA_IMAGE_H = 360
# CARLA_IMAGE_W = 640
# BIRDS_EYE_IMAGE_H = 175 #height of ROI. crops to lane area of carla images
# BIRDS_EYE_IMAGE_W = 640

#SETTINGS FOR 1280x720 CARLA IMAGES:
CARLA_IMAGE_H = 720
CARLA_IMAGE_W = 1280
BIRDS_EYE_IMAGE_H = 350 #height of ROI. crops to lane area of carla images
BIRDS_EYE_IMAGE_W = 1280

H_OFFSET = CARLA_IMAGE_H - BIRDS_EYE_IMAGE_H #offset from top of image to start of ROI
Y_SCALE = 1.429 #7 pixels = length of lane line (10 feet)
X_SCALE = 0.545 #22 pixels = width of lane (12 feet)


class ObjectNode:
    def __init__(self, name, attr, label):
        self.name = name  # Car-1, Car-2.
        self.attr = attr  # bounding box info
        self.label = label # ActorType

    def __repr__(self):
        return "%s" % self.name

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

# {
#    'binary mask':
#    'list of lane marking': [
        # [#1 lane_marking, binary mask]
        # car
        # [#2 lane_marking, binary mask]
#       ]
# }
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
        import pdb; pdb.set_trace()
        type1 = actor1.label
        type2 = actor2.label
        
        low_type = min(type1.value, type2.value) #the lower of the two enums.
        high_type = max(type1.value, type2.value)
    
        function_call = "self.extract_relations_"+ACTOR_NAMES[low_type]+"_"+ACTOR_NAMES[high_type]+"(actor1, actor2)"
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

                
class RealSceneGraph: 
    ''' 
        scene graph the real images 
    '''
    #image_path : path to the image for which the scene graph is generated
    #lane extractor: used to load lane dicts from image directories. Pass None to disable the use of lane information
    def __init__(self, image_path, lane_extractor=None):
        self.g = nx.Graph() #initialize scenegraph as networkx graph

        self.relation_extractor = RelationExtractor()
        self.lane_extractor = lane_extractor
        self.road_node = ObjectNode("Root Road", {}, ActorType.ROAD) # we need to define the type of node.
        #set ego location to middle-bottom of image
        self.ego_node  = ObjectNode("Ego Car", {"location_x": (BIRDS_EYE_IMAGE_W/2) * X_SCALE, "location_y": BIRDS_EYE_IMAGE_H * Y_SCALE}, ActorType.CAR)
        

        self.add_node(self.road_node)   # adding the road as the root node
        self.add_node(self.ego_node)

        # lane/road detection
        if self.lane_extractor != None:
            lanedict = self.lane_extractor.get_lanes_from_file(image_path)
            if lanedict != None:
                #TODO: use pairs of lane lines to add complete lanes instead of lines to the graph
                for lane_line, mask in lanedict.items():
                    lane_line_node = ObjectNode(name="Lane_Marking_" + lane_line, attr=mask, label=ActorType.LANE)
                    self.add_node(lane_line_node)
                    self.add_relation([lane_line_node, Relations.partOf, self.road_node])
                
        # bird eye view projection 
        #warped image is cropped to ROI (contains no sky pixels)
        M = get_birds_eye_matrix()
        warped_img = get_birds_eye_warp(image_path, M) 
        #TODO: map lane lines to warped_img. assign locations to lanes
        #TODO: map vehicles to lanes using locations. add relations to graph

        plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) #plot warped image

        # start detectron2. 
        boxes, labels, image_size = get_bounding_boxes(image_path)

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            box = box.cpu().numpy().tolist()
            class_name = coco_class_names[label]

            if class_name in ['car', 'truck', 'bus']:
                actor_type = ActorType.CAR
            elif class_name in ['person']:
                actor_type = ActorType.PED
            elif class_name in ['bicycle']:
                actor_type = ActorType.BICYCLE
            elif class_name in ['motorcycle']:
                actor_type = ActorType.MOTO
            elif class_name in ['traffic light']:
                actor_type = ActorType.LIGHT
            elif class_name in ['stop sign']:
                actor_type = ActorType.SIGN
        
            attr = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
            print(attr)

            #map center-bottom of bounding box to warped image
            x_mid = (box[2] + box[0]) / 2
            y_bottom = box[3] - H_OFFSET #offset to account for image crop
            pt = np.array([[[x_mid,y_bottom]]], dtype='float32')
            warp_pt = cv2.perspectiveTransform(pt, M)[0][0]

            plt.plot(warp_pt[0], warp_pt[1], color='cyan', marker='o') #plot marked bbox locations
            
            #location/distance in feet
            attr['location_x'] = warp_pt[0] * X_SCALE
            attr['location_y'] = warp_pt[1] * Y_SCALE
            attr['rel_location_x'] = attr['location_x'] - self.ego_node.attr["location_x"]
            attr['rel_location_y'] = attr['location_y'] - self.ego_node.attr["location_y"]
            attr['distance_abs'] = math.sqrt(attr['rel_location_x']**2 + attr['rel_location_y']**2) 

            self.add_node(ObjectNode("%s_%d"%(class_name, idx), attr, actor_type))
            
        plt.show()
        
        # get the relations between nodes
        for node_a, node_b in itertools.combinations(self.g.nodes, 2):
            if node_a.label == ActorType.ROAD or node_b.label == ActorType.ROAD:  
                # dont build relations w/ road
                continue
            
            self.add_relations(self.relation_extractor.extract_relations(node_a, node_b))

        # self.extract_semantic_relations()

    #add single node to graph. node can be any hashable datatype including objects.
    def add_node(self, node):
        self.g.add_node(node, attr=node.attr, label=node.name)
    
    #add relation (edge) between nodes on graph. relation is a list containing [subject, relation, object]
    def add_relation(self, relation):
        if relation != []:
            if relation[0] in self.g.nodes and relation[2] in self.g.nodes:
                self.g.add_edge(relation[0], relation[2], object=relation[1])
            else:
                raise NameError("One or both nodes in relation do not exist in graph. Relation: " + str(relation))
        
    def add_relations(self, relations_list):
        for relation in relations_list:
            self.add_relation(relation)
            
    #parses actor dict and adds nodes to graph. this can be used for all actor types.
    def add_actor_dict(self, actordict):
        for actor_id, attr in actordict.items():
            n = Node(actor_id, attr, None)   #using the actor key as the node name and the dict as its attributes.
            n.name = self.relation_extractor.get_actor_type(n).name.lower() + ":" + actor_id
            n.type = self.relation_extractor.get_actor_type(n).value
            self.add_node(n)
            
    #adds lanes and their dicts. constructs relation between each lane and the root road node.
    def add_lane_dict(self, lanedict):
        n = Node("lane:"+str(lanedict['ego_lane']['lane_id']), lanedict['ego_lane'], ActorType.LANE) 
        self.add_node(n)
        self.add_relation([n, Relations.partOf, self.road_node])
        
        for lane in lanedict['left_lanes']:
            n = Node("lane:"+str(lane['lane_id']), lane, ActorType.LANE)
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])

        for lane in lanedict['right_lanes']:
            n = Node("lane:"+str(lane['lane_id']), lane, ActorType.LANE)
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])
            
    #add signs as entities of the road.
    def add_sign_dict(self, signdict):
        for sign_id, signattr in signdict.items():
            n = Node(sign_id, signattr, ActorType.SIGN)
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])
   
    #calls RelationExtractor to build semantic relations between every pair of entity nodes in graph. call this function after all nodes have been added to graph.
    def extract_semantic_relations(self):
        for node1 in self.g.nodes():
            for node2 in self.g.nodes():
                if node1.name != node2.name: #dont build self-relations
                    if node1.type != ActorType.ROAD.value and node2.type != ActorType.ROAD.value:  # dont build relations w/ road
                        self.add_relations(self.relation_extractor.extract_relations(node1, node2))

def get_bounding_boxes(img_path, out_img_path=None):
    im = cv2.imread(img_path)
    
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # import pdb; pdb.set_trace()

    if out_img_path:
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(out_img_path, out.get_image()[:, :, ::-1])

    return outputs["instances"].pred_boxes, outputs["instances"].pred_classes, outputs["instances"].image_size



#ROI: Region of Interest
#returns transformation matrix for warping image to birds eye projection
#birds eye matrix fixed for all images using the assumption that camera perspective does not change over time.
def get_birds_eye_matrix():
    src = np.float32([[0, BIRDS_EYE_IMAGE_H], [BIRDS_EYE_IMAGE_W, BIRDS_EYE_IMAGE_H], [0, 0], [BIRDS_EYE_IMAGE_W, 0]]) #original dimensions (cropped to ROI)
    dst = np.float32([[int(BIRDS_EYE_IMAGE_W*16/33), BIRDS_EYE_IMAGE_H], [int(BIRDS_EYE_IMAGE_W*17/33), BIRDS_EYE_IMAGE_H], [0, 0], [BIRDS_EYE_IMAGE_W, 0]]) #warped dimensions
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    #Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation (if needed)
    return M
    

#returns image warped to birds eye projection using M
#returned image is vertically cropped to the ROI (lane area)
def get_birds_eye_warp(image_path, M):
    img = cv2.imread(image_path)
    img = img[H_OFFSET:(H_OFFSET+BIRDS_EYE_IMAGE_H), 0:BIRDS_EYE_IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (BIRDS_EYE_IMAGE_W, BIRDS_EYE_IMAGE_H)) # Image warping
    warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB) #set to RGB
    return warped_img


if __name__ == "__main__":
    ##can't use the path on NAS. 
    #　\\128.200.5.40\temp\louisccc\av\synthesis_data\lane-change-804\0\raw_images 00032989.jpg
    le = None #LaneExtractor(r"/home/aung/NAS/louisccc/av/synthesis_data/lane-change-804/0/raw_images")
    realSG = RealSceneGraph(r"/home/aung/NAS/louisccc/av/synthesis_data/lane-change-804/0/raw_images/00032989.jpg", le)
    print(realSG.g.nodes)
    print(realSG.g.edges)

    # get_bounding_boxes(r"/home/aung/NAS/louisccc/av/synthesis_data/lane-change-804/0/raw_images/00032989.jpg", "./00032989.jpg")