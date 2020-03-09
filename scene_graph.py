
import networkx as nx
from relation_extractor import Relations, RelationExtractor
import pdb

#basic class for abstracting a node in the scene graph. this is mainly used for holding the data for each node.
class Node:

    def __init__(self, name, attr):
        self.name = name
        self.attr = attr
    
    
#class defining scene graph and its attributes. contains functions for construction and operations
class SceneGraph:
    
    #graph can be initialized with a framedict to load all objects at once
    def __init__(self, framedict=None):
        self.g = nx.Graph() #initialize scenegraph as networkx graph
        self.relation_extractor = RelationExtractor()
        self.road_node = Node("road", None)
        self.add_node(self.road_node)   #adding the road as the root node
        if framedict != None:
            self.add_frame_dict(framedict)
        
    #add single node to graph. node can be any hashable datatype including objects.
    def add_node(self, node):
        self.g.add_node(node)
        
    #add multiple nodes to graph
    def add_nodes(self, nodes):
        self.g.add_nodes_from(nodes)
    
    #add relation (edge) between nodes on graph. relation is a list containing [subject, relation, object]
    def add_relation(self, relation):
        if relation != []:
            if relation[0] not in self.g.nodes:
                self.add_node(relation[0])
            if relation[2] not in self.g.nodes:
                self.add_node(relation[2])
            self.g.add_edge(relation[0], relation[2], object=relation[1])
        
    def add_relations(self, relations_list):
        for relation in relations_list:
            self.add_relation(relation)
            
    #parses actor dict and adds nodes to graph. this can be used for all actor types.
    def add_actor_dict(self, actordict):
        for actor_id, actorattr in actordict.items():
            n = Node(actor_id, actorattr)   #using the actor key as the node name and the dict as its attributes.
            self.add_node(n)
            
    #adds lanes and their dicts. constructs relation between each lane and the root road node.
    def add_lane_dict(self, lanedict):
        for lane_id, laneattr in lanedict.items():
            n = Node(lane_id, laneattr)
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])
            
    #add the contents of a whole framedict to the graph
    def add_frame_dict(self, framedict):
        for key, attrs in framedict.items():
            if key == "ego":
                self.add_node(Node(key, attrs))
            elif key == "lane":
                self.add_lane_dict(attrs)
            else:
                self.add_actor_dict(attrs)
            
                
            
    #calls RelationExtractor to build semantic relations between every pair of nodes in graph. call this function after all nodes have been added to graph.
    def extract_semantic_relations(self):
        for node1 in self.g.nodes:
            for node2 in self.g.nodes:
                if node1.name != 'road' and node2.name != 'road': #dont build relations with the root road node
                    if node1.name != node2.name: #dont build self-relations
                        self.add_relations(self.relation_extractor.extract_relations(node1.attr, node2.attr))
    

if __name__ == '__main__':
#demo code
    sg = SceneGraph()
    re = RelationExtractor()
    import pdb; pdb.set_trace()

    
    