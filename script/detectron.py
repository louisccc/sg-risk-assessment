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
from pathlib import Path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import detectron2.utils.visualizer 
from detectron2.data import MetadataCatalog
# panoptic seg
sys.path.append(os.path.dirname(sys.path[0]))
from core.relation_extractor import ActorType, Relations
#from core.lane_extractor import LaneExtractor #(not currently used)

from enum import Enum

#SETTINGS FOR 1280x720 CARLA IMAGES:
CARLA_IMAGE_H = 720
CARLA_IMAGE_W = 1280
BIRDS_EYE_IMAGE_H = 350 #height of ROI. crops to lane area of carla images
BIRDS_EYE_IMAGE_W = 1280

H_OFFSET = CARLA_IMAGE_H - BIRDS_EYE_IMAGE_H #offset from top of image to start of ROI
Y_SCALE = 1.429 #7 pixels = length of lane line (10 feet)
X_SCALE = 0.545 #22 pixels = width of lane (12 feet)

CAR_PROXIMITY_THRESH_SUPER_NEAR = 50 # max number of feet between a car and another entity to build proximity relation
CAR_PROXIMITY_THRESH_VERY_NEAR = 150
CAR_PROXIMITY_THRESH_NEAR = 300
CAR_PROXIMITY_THRESH_VISIBLE = 500

LANE_THRESHOLD = 6 #feet. if object's center is more than this distance away from ego's center, build left or right lane relation
CENTER_LANE_THRESHOLD = 9 #feet. if object's center is within this distance of ego's center, build middle lane relation

def create_text_labels_with_idx(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{}_{} {:.0f}%".format(l, idx, s * 100) for idx, (l, s) in enumerate(zip(labels, scores))]
    return labels

detectron2.utils.visualizer._create_text_labels = create_text_labels_with_idx

class ObjectNode:
    def __init__(self, name, attr, label):
        self.name = name  # Car-1, Car-2.
        self.attr = attr  # bounding box info
        self.label = label # ActorType

    def __repr__(self):
        return "%s" % self.name  

                
class RealSceneGraph: 
    ''' 
        scene graph the real images 
    '''
    #image_path : path to the image for which the scene graph is generated
    #lane extractor: used to load lane dicts from image directories. Pass None to disable the use of lane information
    def __init__(self, image_path, lane_extractor=None):
        self.g = nx.Graph() #initialize scenegraph as networkx graph

        self.lane_extractor = lane_extractor
        self.road_node = ObjectNode("Root Road", {}, ActorType.ROAD) # we need to define the type of node.
        #set ego location to middle-bottom of image
        self.ego_node  = ObjectNode("Ego Car", {"location_x": (BIRDS_EYE_IMAGE_W/2) * X_SCALE, "location_y": BIRDS_EYE_IMAGE_H * Y_SCALE}, ActorType.CAR)
        self.add_node(self.road_node)   # adding the road as the root node
        self.add_node(self.ego_node)


        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        self.coco_class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
        
        self.predictor = DefaultPredictor(self.cfg)

                
        # bird eye view projection 
        #warped image is cropped to ROI (contains no sky pixels)
        M = get_birds_eye_matrix()
        warped_img = get_birds_eye_warp(image_path, M) 
        #TODO: map lane lines to warped_img. assign locations to lanes
        #TODO: map vehicles to lanes using locations. add relations to graph

        plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) #plot warped image

        # start detectron2. 
        out_img_path = Path(image_path).resolve().parent.parent / "obj_det_results"
        out_img_path.mkdir(exist_ok=True)
        boxes, labels, image_size = self.get_bounding_boxes(image_path, str(out_img_path / str(Path(image_path).name)))

        for idx, (box, label) in enumerate(zip(boxes, labels)):
            box = box.cpu().numpy().tolist()
            class_name = self.coco_class_names[label]

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
            else:
                continue
        
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
            node = ObjectNode("%s_%d"%(class_name, idx), attr, actor_type)
            self.add_node(node)
            self.map_to_relative_lanes(node)

        plt.show()
        
        # get the relations between nodes
        for node_a, node_b in itertools.combinations(self.g.nodes, 2):
            relation_list = []
            if node_a.label == ActorType.ROAD or node_b.label == ActorType.ROAD:  
                # dont build relations w/ road
                continue
            if node_a.label == ActorType.CAR and node_b.label == ActorType.CAR:
                relation_list += self.create_proximity_relations(node_a, node_b)
                relation_list += self.create_directional_relations(node_a, node_b)
                self.add_relations(relation_list)
            import pdb; pdb.set_trace()

    def create_proximity_relations(self, actor1, actor2):
        if self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_SUPER_NEAR:
            return [[actor1, Relations.super_near, actor2]]
        elif self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VERY_NEAR:
            return [[actor1, Relations.very_near, actor2]]
        elif self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_NEAR:
            return [[actor1, Relations.near, actor2]]
        elif self.euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VISIBLE:
            return [[actor1, Relations.visible, actor2]]
        return []

    def euclidean_distance(self, actor1, actor2):
        #import pdb; pdb.set_trace()
        l1 = (actor1.attr['location_x'], actor1.attr['location_y'])
        l2 = (actor2.attr['location_x'], actor2.attr['location_y'])
        return math.sqrt((l1[0] - l2[0])**2 + (l1[1]- l2[1])**2)

    def create_directional_relations(self, actor1, actor2):
        relation_list = []

        # actor2 is in front of actor1
        if actor2.attr['location_y'] < actor1.attr['location_y']:
            if abs(actor2.attr['location_x'] - actor1.attr['location_x']) <= 12:
                relation_list.append([actor1, Relations.front, actor2])
            # actor2 to the left of actor1 
            elif actor2.attr['location_x'] < actor1.attr['location_x']:
                relation_list.append([actor1, Relations.frontLeft, actor2])
            # actor2 to the right of actor1 
            elif actor2.attr['location_x'] > actor1.attr['location_x']:
                relation_list.append([actor1, Relations.frontRight, actor2])
                
        # actor2 is behind actor1
        else:
            # actor2 is directly behind of actor1
            if  abs(actor2.attr['location_x'] - actor1.attr['location_x']) <= 12:
                relation_list.append([actor1, Relations.rear, actor2])
            # actor2 to the left of actor1 
            elif actor2.attr['location_x'] < actor1.attr['location_x']:
                relation_list.append([actor1, Relations.rearLeft, actor2])
             # actor2 to the left of actor1 
            elif actor2.attr['location_x'] > actor1.attr['location_x']:
                relation_list.append([actor1, Relations.rearRight, actor2])

        ### disable rear relations help the inference. 
        return relation_list

    # lane/road detection using LaneNet (not currently used)
    def extract_lanenet_lanes(self, image_path):
        if self.lane_extractor != None:
            lanedict = self.lane_extractor.get_lanes_from_file(image_path)
            if lanedict != None:
                #TODO: use pairs of lane lines to add complete lanes instead of lines to the graph
                for lane_line, mask in lanedict.items():
                    lane_line_node = ObjectNode(name="Lane_Marking_" + lane_line, attr=mask, label=ActorType.LANE)
                    self.add_node(lane_line_node)
                    self.add_relation([lane_line_node, Relations.partOf, self.road_node])

    #relative lane mapping method. Each vehicle is assigned to left, middle, or right lane depending on relative position to ego
    def extract_relative_lanes(self):
        self.left_lane = ObjectNode("Left Lane", {}, ActorType.LANE)
        self.right_lane = ObjectNode("Right Lane", {}, ActorType.LANE)
        self.middle_lane = ObjectNode("Middle Lane", {}, ActorType.LANE)
        self.add_node(self.left_lane)
        self.add_node(self.right_lane)
        self.add_node(self.middle_lane)
        self.add_relation([self.left_lane, Relations.partOf, self.road_node])
        self.add_relation([self.right_lane, Relations.partOf, self.road_node])
        self.add_relation([self.middle_lane, Relations.partOf, self.road_node])
        self.add_relation([self.ego_node, Relations.isIn, self.middle_lane])

    #builds isIn relation between object and lane depending on x-displacement relative to ego
    #left/middle and right/middle relations have an overlap area determined by the size of CENTER_LANE_THRESHOLD and LANE_THRESHOLD.
    #TODO: move to relation_extractor in replacement of current lane-vehicle relation code
    def map_to_relative_lanes(self, object_node):
        if object_node.label in [ActorType.LANE, ActorType.LIGHT, ActorType.SIGN, ActorType.ROAD]: #don't build lane relations with static objects
            return
        if object_node.attr['rel_location_x'] < -LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.left_lane])
        elif object_node.attr['rel_location_x'] > LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.right_lane])
        if object_node.attr['rel_location_x'] > -CENTER_LANE_THRESHOLD and object_node.attr['rel_location_x'] < CENTER_LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.middle_lane])

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
            
    ###DEPRECATED###. TODO: remove if not needed
    # #adds lanes and their dicts. constructs relation between each lane and the root road node.
    # def add_lane_dict(self, lanedict):
    #     n = Node("lane:"+str(lanedict['ego_lane']['lane_id']), lanedict['ego_lane'], ActorType.LANE) 
    #     self.add_node(n)
    #     self.add_relation([n, Relations.partOf, self.road_node])
        
    #     for lane in lanedict['left_lanes']:
    #         n = Node("lane:"+str(lane['lane_id']), lane, ActorType.LANE)
    #         self.add_node(n)
    #         self.add_relation([n, Relations.partOf, self.road_node])

    #     for lane in lanedict['right_lanes']:
    #         n = Node("lane:"+str(lane['lane_id']), lane, ActorType.LANE)
    #         self.add_node(n)
    #         self.add_relation([n, Relations.partOf, self.road_node])
            
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

    def get_bounding_boxes(self, img_path, out_img_path=None):
        im = cv2.imread(img_path)
        outputs = self.predictor(im)

        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        
        # import pdb; pdb.set_trace()

        if out_img_path:
            # We can use `Visualizer` to draw the predictions on the image.
            v = detectron2.utils.visualizer.Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
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
    realSG = RealSceneGraph(r"/home/aung/NAS/louisccc/av/synthesis_data/new_recording_2/3/raw_images/00016995.jpg", le)
    print(realSG.g.nodes)
    print(realSG.g.edges)
    # get_bounding_boxes(r"/home/aung/NAS/louisccc/av/synthesis_data/lane-change-804/0/raw_images/00032989.jpg", "./00032989.jpg")