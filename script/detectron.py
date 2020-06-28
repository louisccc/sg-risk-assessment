# -*- coding: utf-8 -*-
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import sys, os
import numpy as np
import cv2
import random
import networkx as nx

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

sys.path.append(os.path.dirname(sys.path[0]))
from core.relation_extractor import ActorType, RelationExtractor

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



class ObjectNode:
    def __init__(self, name, attr, label):
        self.name = name  # Car-1, Car-2.
        self.attr = attr  # bounding box info
        self.label = name # ActorType

    def __repr__(self):
        return "%s" % self.name
        
class RealSceneGraph: 
    ''' 
        scene graph the real images 
    '''
    
    def __init__(self, image_path):
        self.g = nx.Graph() #initialize scenegraph as networkx graph

        self.relation_extractor = RelationExtractor()
        
        self.road_node = ObjectNode("Root Road", {}, ActorType.ROAD) # we need to define the type of node.
        self.ego_node  = ObjectNode("Ego Car", {}, ActorType.CAR)
        # we need to fake a preset position of ego car. 

        self.add_node(self.road_node)   #adding the road as the root node
        self.add_node(self.ego_node)

        # start detectron2. 
        boxes, labels = get_bounding_boxes(image_path)

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

            self.add_node(ObjectNode("%s_%d"%(class_name, idx), attr, actor_type))
            import pdb; pdb.set_trace()
            
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
        n = Node("lane:"+str(lanedict['ego_lane']['lane_id']), lanedict['ego_lane'], ActorType.LANE) #todo: change to true when lanedict entry is complete
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

    return outputs["instances"].pred_boxes, outputs["instances"].pred_classes

if __name__ == "__main__":
    ##can't use the path on NAS. 
    realSG = RealSceneGraph(r"../input/synthesis_data/lane-change/0/raw_images/00025857.png")