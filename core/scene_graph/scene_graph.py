import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from .relation_extractor import Relations, ActorType, RelationExtractor
import pdb, json, random
from pathlib import Path
from glob import glob
import pickle as pkl
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import torch
#basic class for abstracting a node in the scene graph. this is mainly used for holding the data for each node.

class Node:
    def __init__(self, name, attr, type=None, is_entity=False):
        self.name = name
        self.attr = attr
        self.label = name #+ "\n" + str(attr)
        self.is_entity = is_entity
        self.type = type.value if type != None else None

    def __repr__(self):
        return "%s" % self.name


#class defining scene graph and its attributes. contains functions for construction and operations
class SceneGraph:
    
    #graph can be initialized with a framedict to load all objects at once
    def __init__(self, framedict):
        self.g = nx.Graph() #initialize scenegraph as networkx graph
                
        self.relation_extractor = RelationExtractor()
        
        self.road_node = Node("Root Road", {}, ActorType.ROAD)
                
        self.add_node(self.road_node)   #adding the road as the root node

        self.parse_json(framedict) # processing json framedict:

    #add single node to graph. node can be any hashable datatype including objects.
    def add_node(self, node):
        self.g.add_node(node, attr=node.attr, label=node.name)
    
    #add relation (edge) between nodes on graph. relation is a list containing [subject, relation, object]
    def add_relation(self, relation):
        if relation != []:
            if relation[0] not in self.g.nodes:
                self.add_node(relation[0])
            if relation[2] not in self.g.nodes:
                self.add_node(relation[2])
            if relation[0] in self.g.nodes and relation[2] in self.g.nodes:
                self.g.add_edge(relation[0], relation[2], object=relation[1])
        
    def add_relations(self, relations_list):
        for relation in relations_list:
            self.add_relation(relation)
            
    #parses actor dict and adds nodes to graph. this can be used for all actor types.
    def add_actor_dict(self, actordict):
        for actor_id, attr in actordict.items():
            n = Node(actor_id, attr, None, True)   #using the actor key as the node name and the dict as its attributes.
            n.type = self.relation_extractor.get_actor_type(n).value
            self.add_node(n)
            self.add_attributes(n, attr)
            
    #adds lanes and their dicts. constructs relation between each lane and the root road node.
    def add_lane_dict(self, lanedict):
        
        n = Node("lane:"+str(lanedict['ego_lane']['lane_id']), lanedict['ego_lane'], ActorType.LANE) #todo: change to true when lanedict entry is complete
        self.add_node(n)
        self.add_relation([n, Relations.partOf, self.road_node])
        
        for lane in lanedict['left_lanes']:
            # for lane_id, laneattr in lane.items():
            n = Node("lane:"+str(lane['lane_id']), lane, ActorType.LANE) #todo: change to true when lanedict entry is complete
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])

        for lane in lanedict['right_lanes']:
            n = Node("lane:"+str(lane['lane_id']), lane, ActorType.LANE) #todo: change to true when lanedict entry is complete
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])
            
    #add signs as entities of the road.
    def add_sign_dict(self, signdict):
        for sign_id, signattr in signdict.items():
            n = Node(sign_id, signattr, ActorType.SIGN, True)
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])

    #add the contents of a whole framedict to the graph
    def parse_json(self, framedict):
        
        for key, attrs in framedict.items():   
            if key == "ego":
                egoNode = Node(key+":"+attrs['name'], attrs, ActorType.CAR)
                self.add_node(egoNode)
                # self.add_attributes(egoNode, attrs)
            elif key == "lane":
                self.add_lane_dict(attrs)
            elif key == "sign":
                self.add_sign_dict(attrs)
            elif key == "actors":
                # parsing a list of actors
                for actor_id, attr in attrs.items():
                    n = Node(actor_id, attr, None)   #using the actor key as the node name and the dict as its attributes.
                    n.name = self.relation_extractor.get_actor_type(n).name.lower() + ":" + actor_id
                    n.type = self.relation_extractor.get_actor_type(n).value
                    self.add_node(n)
        
        self.extract_semantic_relations()
            
    def extract_semantic_relations(self):
        #calls RelationExtractor to build semantic relations between every pair of entity nodes in graph. call this function after all nodes have been added to graph.
        for node1 in self.g.nodes():
            for node2 in self.g.nodes():
                if node1.name != node2.name: #dont build self-relations
                    if node1.type != ActorType.ROAD.value and node2.type != ActorType.ROAD.value:  # dont build relations w/ road
                        self.add_relations(self.relation_extractor.extract_relations(node1, node2))

    def visualize(self):
        pos = nx.spring_layout(self.g, k=1.5*1/np.sqrt(len(self.g.nodes())))

        color_map = []
        for node in self.g.nodes():
            if node.type == ActorType.ROAD.value:
                color_map.append(1)
            elif node.type == ActorType.LANE.value:
                color_map.append(3)
            else:
                color_map.append(2)

        nx.draw(self.g, node_color=color_map, labels=nx.get_node_attributes(self.g, 'label'), pos=pos, font_size=8, with_labels=True)
        plt.show()