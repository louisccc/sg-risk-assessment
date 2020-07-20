import matplotlib, math, itertools
matplotlib.use("Agg")
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from core.relation_extractor import Relations, ActorType, RelationExtractor, RELATION_COLORS


LANE_THRESHOLD = 6 #feet. if object's center is more than this distance away from ego's center, build left or right lane relation
CENTER_LANE_THRESHOLD = 9 #feet. if object's center is within this distance of ego's center, build middle lane relation


#class representing a node in the scene graph. this is mainly used for holding the data for each node.
class Node:
    def __init__(self, name, attr, type=None):
        self.name = name
        self.attr = attr
        self.label = name
        self.type = type.value if type != None else None

    def __repr__(self):
        return "%s" % self.name


#class defining scene graph and its attributes. contains functions for construction and operations
class SceneGraph:
    
    #graph can be initialized with a framedict to load all objects at once
    def __init__(self, framedict, framenum=None):
        self.g = nx.MultiDiGraph() #initialize scenegraph as networkx graph
        self.road_node = Node("Root Road", {}, ActorType.ROAD)
        self.add_node(self.road_node)   #adding the road as the root node
        self.parse_json(framedict) # processing json framedict

    #add single node to graph. node can be any hashable datatype including objects.
    def add_node(self, node):
        color = "white"
        if node.name.startswith("ego"):
            color = "red"
        elif node.name.startswith("car"):
            color = "blue"
        elif node.name.startswith("lane"):
            color = "yellow"
        self.g.add_node(node, attr=node.attr, label=node.name, style='filled', fillcolor=color)
    
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
            
    #parses actor dict and adds nodes to graph. this can be used for all actor types.
    def add_actor_dict(self, actordict):
        for actor_id, attr in actordict.items():
            # filter actors behind ego 
            x1, y1 = math.cos(math.radians(self.egoNode.attr['rotation'][0])), math.sin(math.radians(self.egoNode.attr['rotation'][0]))
            x2, y2 = attr['location'][0] - self.egoNode.attr['location'][0], attr['location'][1] - self.egoNode.attr['location'][1]
            inner_product = x1*x2 + y1*y2
            length_product = math.sqrt(x1**2+y1**2) + math.sqrt(x2**2+y2**2)
            degree = math.degrees(math.acos(inner_product / length_product))
            
            if degree <= 80 or (degree >=280 and degree <= 360):
                # if abs(self.egoNode.attr['lane_idx'] - attr['lane_idx']) <= 1 \
                # or ("invading_lane" in self.egoNode.attr and (2*self.egoNode.attr['invading_lane'] - self.egoNode.attr['orig_lane_idx']) == attr['lane_idx']):
                n = Node(actor_id, attr, None)   #using the actor key as the node name and the dict as its attributes.
                n.name = self.relation_extractor.get_actor_type(n).name.lower() + ":" + actor_id
                n.type = self.relation_extractor.get_actor_type(n).value
                self.add_node(n)
            
    #adds lanes and their dicts. constructs relation between each lane and the root road node.
    def add_lane_dict(self, lanedict):
        #TODO: can we filter out the lane that has no car on it?
        for idx, lane in enumerate(lanedict['lanes']):
            lane['lane_idx'] = idx
            n = Node("lane:"+str(idx), lane, ActorType.LANE)
            self.add_node(n)
            self.add_relation([n, Relations.isIn, self.road_node])
            
    #add signs as entities of the road.
    def add_sign_dict(self, signdict):
        for sign_id, signattr in signdict.items():
            n = Node(sign_id, signattr, ActorType.SIGN)
            self.add_node(n)
            self.add_relation([n, Relations.isIn, self.road_node])

    #add the contents of a whole framedict to the graph
    def parse_json(self, framedict):
        self.egoNode = Node("ego:"+framedict['ego']['name'], framedict['ego'], ActorType.CAR)
        self.add_node(self.egoNode)

        #rotating axes to align with ego. yaw axis is the primary rotation axis in vehicles
        self.ego_yaw = math.radians(self.egoNode.attr['rotation'][0])
        self.ego_cos_term = math.cos(self.ego_yaw)
        self.ego_sin_term = math.sin(self.ego_yaw)

        self.relation_extractor = RelationExtractor(self.egoNode)
        # self.add_attributes(egoNode, attrs)
        for key, attrs in framedict.items():   
            if key == "lane":
                self.add_lane_dict(attrs)
            elif key == "sign":
                self.add_sign_dict(attrs)
            elif key == "actors":
                self.add_actor_dict(attrs)
        self.extract_semantic_relations()
    
    #calls RelationExtractor to build semantic relations between every pair of entity nodes in graph. call this function after all nodes have been added to graph.
    def extract_semantic_relations(self):
        for node1, node2 in itertools.combinations(self.g.nodes, 2):
            if node1.name != node2.name: #dont build self-relations
                if node1.type != ActorType.ROAD.value and node2.type != ActorType.ROAD.value:  # dont build relations w/ road
                    self.add_relations(self.relation_extractor.extract_relations(node1, node2))

    def visualize(self, filename=None):
        A = to_agraph(self.g)
        A.layout('dot')
        A.draw(filename)

    
    #TODO refactor after testing
    #relative lane mapping method. Each vehicle is assigned to left, middle, or right lane depending on relative position to ego
    def extract_relative_lanes(self):
        self.left_lane = Node("lane_left", {}, ActorType.LANE)
        self.right_lane = Node("lane_right", {}, ActorType.LANE)
        self.middle_lane = Node("lane_middle", {}, ActorType.LANE)
        self.add_node(self.left_lane)
        self.add_node(self.right_lane)
        self.add_node(self.middle_lane)
        self.add_relation([self.left_lane, Relations.isIn, self.road_node])
        self.add_relation([self.right_lane, Relations.isIn, self.road_node])
        self.add_relation([self.middle_lane, Relations.isIn, self.road_node])
        self.add_relation([self.egoNode, Relations.isIn, self.middle_lane])

    #builds isIn relation between object and lane depending on x-displacement relative to ego
    #left/middle and right/middle relations have an overlap area determined by the size of CENTER_LANE_THRESHOLD and LANE_THRESHOLD.
    #TODO: move to relation_extractor in replacement of current lane-vehicle relation code
    def add_mapping_to_relative_lanes(self, object_node):
        if object_node.label in [ActorType.LANE, ActorType.LIGHT, ActorType.SIGN, ActorType.ROAD]: #don't build lane relations with static objects
            return
        _, ego_y = self.rotate_coords(self.egoNode.attr['location'][0], self.egoNode.attr['location'][1]) #NOTE: X corresponds to forward/back displacement and Y corresponds to left/right displacement
        _, new_y = self.rotate_coords(object_node.attr['location'][0], object_node.attr['location'][1])
        y_diff = new_y - ego_y
        if y_diff < -LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.left_lane])
        elif y_diff > LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.right_lane])
        if abs(y_diff) <= CENTER_LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.middle_lane])
    

    #copied from get_node_embeddings(). rotates coordinates to be relative to ego vector.
    def rotate_coords(self, x, y): 
        new_x = (x*self.ego_cos_term) + (y*self.ego_sin_term)
        new_y = ((-x)*self.ego_sin_term) + (y*self.ego_cos_term)
        return new_x, new_y