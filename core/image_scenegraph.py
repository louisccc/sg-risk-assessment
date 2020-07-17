# -*- coding: utf-8 -*-
import sys, os, cv2, itertools, math, matplotlib
matplotlib.use("Agg")
sys.path.append(os.path.dirname(sys.path[0]))
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph
from core.relation_extractor import ActorType, Relations, RELATION_COLORS
# from core.lane_extractor import LaneExtractor


#SETTINGS FOR 1280x720 CARLA IMAGES:
CARLA_IMAGE_H = 720
CARLA_IMAGE_W = 1280
CROPPED_H = 350 #height of ROI. crops to lane area of carla images
BIRDS_EYE_IMAGE_H = 850 
BIRDS_EYE_IMAGE_W = 1280

H_OFFSET = CARLA_IMAGE_H - CROPPED_H #offset from top of image to start of ROI
Y_SCALE = 0.55 #18 pixels = length of lane line (10 feet)
X_SCALE = 0.54 #22 pixels = width of lane (12 feet)

CAR_PROXIMITY_THRESH_SUPER_NEAR = 50 # max number of feet between a car and another entity to build proximity relation
CAR_PROXIMITY_THRESH_VERY_NEAR = 150
CAR_PROXIMITY_THRESH_NEAR = 300
CAR_PROXIMITY_THRESH_VISIBLE = 500

LANE_THRESHOLD = 6 #feet. if object's center is more than this distance away from ego's center, build left or right lane relation
CENTER_LANE_THRESHOLD = 9 #feet. if object's center is within this distance of ego's center, build middle lane relation


class ObjectNode:
    def __init__(self, name, attr, label):
        self.name = name  # Car-1, Car-2.
        self.attr = attr  # bounding box info
        self.label = label # ActorType

    def __repr__(self):
        return "%s" % (self.name)

                
class RealSceneGraph: 
    ''' 
        scene graph the real images 
        arguments: 
            image_path : path to the image for which the scene graph is generated
            lane extractor: used to load lane dicts from image directories. Pass None to disable the use of lane information
    '''
    def __init__(self, image_path, bounding_boxes, coco_class_names=None, lane_extractor=None):
        self.g = nx.MultiDiGraph() #initialize scenegraph as networkx graph

        # road and lane settings.
        self.lane_extractor = lane_extractor
        self.road_node = ObjectNode("Root Road", {}, ActorType.ROAD) # we need to define the type of node.
        self.add_node(self.road_node)   # adding the road as the root node
        
        # set ego location to middle-bottom of image.
        self.ego_location = ((BIRDS_EYE_IMAGE_W/2) * X_SCALE, BIRDS_EYE_IMAGE_H * Y_SCALE)
        self.ego_node  = ObjectNode("Ego Car", {"location_x": self.ego_location[0], "location_y": self.ego_location[1]}, ActorType.CAR)
        self.add_node(self.ego_node)
        
        self.extract_relative_lanes() ### three lane formulation.
        
        boxes, labels, image_size = bounding_boxes

        ### TODO: Arnav's part lane/road detection
        if self.lane_extractor != None:
            lanedict = self.lane_extractor.get_lanes_from_file(image_path)
            if lanedict != None:
                #TODO: use pairs of lane lines to add complete lanes instead of lines to the graph
                for lane_line, mask in lanedict.items():
                    lane_line_node = ObjectNode(name="Lane_Marking_" + lane_line, attr=mask, label=ActorType.LANE)
                    self.add_node(lane_line_node)
                    self.add_relation([lane_line_node, Relations.partOf, self.road_node])
                
        # bird eye view projection
        # warped image is cropped to ROI (contains no sky pixels)
        M = get_birds_eye_matrix()
        warped_img = get_birds_eye_warp(image_path, M) 
        #TODO: map lane lines to warped_img. assign locations to lanes
        #TODO: map vehicles to lanes using locations. add relations to graph

        cv2.imwrite( "./warped.jpg", cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) #plot warped image
        ### TODO: Arnav's part lane/road detection

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
            else:
                continue
        
            attr = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}

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
            self.add_mapping_to_relative_lanes(node)

        # get the relations between nodes
        for node_a, node_b in itertools.combinations(self.g.nodes, 2):
            relation_list = []
            if node_a.label == ActorType.ROAD or node_b.label == ActorType.ROAD:  
                # dont build relations w/ road
                continue
            if node_a.label == ActorType.CAR and node_b.label == ActorType.CAR:
                relation_list += self.extract_proximity_relations(node_a, node_b)
                relation_list += self.extract_directional_relations(node_a, node_b)
                relation_list += self.extract_proximity_relations(node_b, node_a)
                relation_list += self.extract_directional_relations(node_b, node_a)
                self.add_relations(relation_list)
    
    def extract_proximity_relations(self, actor1, actor2):
        if   self.get_euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_SUPER_NEAR:
            return [[actor1, Relations.super_near, actor2]]
        elif self.get_euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VERY_NEAR:
            return [[actor1, Relations.very_near, actor2]]
        elif self.get_euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_NEAR:
            return [[actor1, Relations.near, actor2]]
        elif self.get_euclidean_distance(actor1, actor2) < CAR_PROXIMITY_THRESH_VISIBLE:
            return [[actor1, Relations.visible, actor2]]
        return []

    def get_euclidean_distance(self, actor1, actor2):
        l1 = (actor1.attr['location_x'], actor1.attr['location_y'])
        l2 = (actor2.attr['location_x'], actor2.attr['location_y'])
        return math.sqrt((l1[0] - l2[0])**2 + (l1[1]- l2[1])**2)

    def extract_directional_relations(self, actor1, actor2):
        relation_list = []

        # actor2 is in front of actor1
        if actor2.attr['location_y'] < actor1.attr['location_y']:
            relation_list.append([actor2, Relations.inFrontOf, actor1])
        # actor2 is behind actor1
        else:
            # actor2 is behind actor1
            relation_list.append([actor2, Relations.atRearOf, actor1])
        
        if abs(actor2.attr['location_x'] - actor1.attr['location_x']) <= CENTER_LANE_THRESHOLD:
                pass
        # actor2 to the left of actor1 
        elif actor2.attr['location_x'] < actor1.attr['location_x']:
            relation_list.append([actor2, Relations.toLeftOf, actor1])
        # actor2 to the right of actor1 
        elif actor2.attr['location_x'] > actor1.attr['location_x']:
            relation_list.append([actor2, Relations.toRightOf, actor1])
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
        self.add_relation([self.left_lane, Relations.isIn, self.road_node])
        self.add_relation([self.right_lane, Relations.isIn, self.road_node])
        self.add_relation([self.middle_lane, Relations.isIn, self.road_node])
        self.add_relation([self.ego_node, Relations.isIn, self.middle_lane])

    #builds isIn relation between object and lane depending on x-displacement relative to ego
    #left/middle and right/middle relations have an overlap area determined by the size of CENTER_LANE_THRESHOLD and LANE_THRESHOLD.
    #TODO: move to relation_extractor in replacement of current lane-vehicle relation code
    def add_mapping_to_relative_lanes(self, object_node):
        if object_node.label in [ActorType.LANE, ActorType.LIGHT, ActorType.SIGN, ActorType.ROAD]: #don't build lane relations with static objects
            return
        if object_node.attr['rel_location_x'] < -LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.left_lane])
        elif object_node.attr['rel_location_x'] > LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.right_lane])
        if abs(object_node.attr['rel_location_x']) <= CENTER_LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.middle_lane])

    #add single node to graph. node can be any hashable datatype including objects.
    def add_node(self, node):
        color = "white"
        if node.name.startswith("ego"):
            color = "red"
        elif node.name.startswith("car"):
            color = "blue"
        elif node.name.startswith("lane"):
            color = "yellow"
        self.g.add_node(node, attr=node.attr, label=node.name,  style='filled', fillcolor=color)
    
    #add relation (edge) between nodes on graph. relation is a list containing [subject, relation, object]
    def add_relation(self, relation):
        if relation != []:
            if relation[0] in self.g.nodes and relation[2] in self.g.nodes:
                self.g.add_edge(relation[0], relation[2], object=relation[1], label=relation[1].name, color=RELATION_COLORS[int(relation[1].value)])
            else:
                raise NameError("One or both nodes in relation do not exist in graph. Relation: " + str(relation))
        
    def add_relations(self, relations_list):
        for relation in relations_list:
            self.add_relation(relation)
            
    def visualize(self, to_filename):
        A = to_agraph(self.g)
        A.layout('dot')
        A.draw(to_filename)

#ROI: Region of Interest
#returns transformation matrix for warping image to birds eye projection
#birds eye matrix fixed for all images using the assumption that camera perspective does not change over time.
def get_birds_eye_matrix():
    src = np.float32([[0, CROPPED_H], [BIRDS_EYE_IMAGE_W, CROPPED_H], [0, 0], [BIRDS_EYE_IMAGE_W, 0]]) #original dimensions (cropped to ROI)
    dst = np.float32([[int(BIRDS_EYE_IMAGE_W*16/33), BIRDS_EYE_IMAGE_H], [int(BIRDS_EYE_IMAGE_W*17/33), BIRDS_EYE_IMAGE_H], [0, 0], [BIRDS_EYE_IMAGE_W, 0]]) #warped dimensions
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    #Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation (if needed)
    return M
    

#returns image warped to birds eye projection using M
#returned image is vertically cropped to the ROI (lane area)
def get_birds_eye_warp(image_path, M):
    img = cv2.imread(image_path)
    img = img[H_OFFSET:CARLA_IMAGE_H, 0:BIRDS_EYE_IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (BIRDS_EYE_IMAGE_W, BIRDS_EYE_IMAGE_H)) # Image warping
    warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB) #set to RGB
    return warped_img