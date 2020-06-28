import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import math
import torch
import json

from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pygcn.utils import sparse_mx_to_torch_sparse_tensor, normalize, accuracy
import scipy.sparse as sp
import torch.nn.functional as F
from pathlib import Path 
from tqdm import tqdm 
from .relation_extractor import Relations, ActorType, RelationExtractor

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
    def __init__(self, framedict):
        self.g = nx.Graph() #initialize scenegraph as networkx graph
        self.relation_extractor = RelationExtractor()
        self.road_node = Node("Root Road", {}, ActorType.ROAD)
        self.add_node(self.road_node)   #adding the road as the root node
        self.parse_json(framedict) # processing json framedict

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
                self.add_actor_dict(attrs)
        self.extract_semantic_relations()
    
    #calls RelationExtractor to build semantic relations between every pair of entity nodes in graph. call this function after all nodes have been added to graph.
    def extract_semantic_relations(self):
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


class SceneGraphSequenceGenerator:
    def __init__(self):
        self.scenegraphs_sequence = []
        self.cache_filename = 'dyngraph_embeddings.pkl'
        
        # config used for parsing CARLA:
        # this is the number of global classes defined in CARLA.
        self.num_classes = 8
        
        # gets a list of all feature labels (which will be used) for all scenegraphs
        self.feature_list = {"rel_location_x", 
                             "rel_location_y", 
                             "rel_location_z", #add 3 columns for relative vector values
                             "distance_abs", # adding absolute distance to ego
                            }
        # create 1hot class labels columns.
        for i in range(self.num_classes):
            self.feature_list.add("type_"+str(i))

    def load(self, input_path):
        scenegraphs = {}

        for path in tqdm([x for x in input_path.iterdir() if x.is_dir()]):

            for txt_path in glob("%s/**/*.txt" % str(path/"scene_raw"), recursive=True):
                scene_dict_f = open(txt_path, 'r')
                try:
                    framedict = json.loads(scene_dict_f.read())

                    for frame, frame_dict in framedict.items():
                        scenegraph = SceneGraph(frame_dict)
                        scenegraphs[frame] = scenegraph

                except Exception as e:
                    print("We have problem parsing the dict.json in %s"%txt_path)
                    print(e)
                

            label_f = open(str(path/"label.txt"), 'r')
            risk_label = int(label_f.read())

            if risk_label >= 0:
                risk_label = 1
            else:
                risk_label = 0

            self.scenegraphs_sequence.append((scenegraphs, risk_label))

    def cache_exists(self):
        return Path(self.cache_filename).exists()
            
    def process_graph_sequences(self, number_of_frames):
        self.subsampled_sequences = self.subsample(number_of_frames=20)
        self.scenegraphs_sequence = self.subsampled_sequences

        '''
            The self.scenegraphs_sequence should be having same length after the subsampling. 
            This function will get the graph-related features (node embeddings, edge types, adjacency matrix) from scenegraphs.
            in tensor formats.
        '''

        for scenegraphs, _ in self.scenegraphs_sequence:
            for scenegraph in scenegraphs:
                node_name2idx = {node:idx for idx, node in enumerate(scenegraph.g.nodes)}

                scenegraph.node_features                    = self.get_node_embeddings(scenegraph)
                scenegraph.edge_index, scenegraph.edge_attr = self.get_edge_embeddings(scenegraph, node_name2idx)

    def subsample(self, number_of_frames=20): 
        '''
            This functions will subsample the original scenegraph sequence dataset (self.scenegraphs_sequence). 
            Before running this function, it includes a variant length of graph sequences. 
            We expect the length of graph sequences will be homogenenous after running this function.

            The default value of number_of_frames will be 20; Could be a tunnable hyperparameters.
        '''
        
        sequences = [] 
        sequence_labels = []

        for scenegraphs, risk_label in self.scenegraphs_sequence:
            sequence = []
            acc_number = 0
            modulo = int(len(scenegraphs) / number_of_frames)
            for idx, (timeframe, scenegraph) in enumerate(scenegraphs.items()):
                if idx % modulo == 0 and acc_number < number_of_frames:
                    sequence.append(scenegraph)
                    acc_number+=1
            
            sequences.append(sequence)
            sequence_labels.append(risk_label)

        subsampled_list = list(zip(sequences, sequence_labels))
        return subsampled_list
        
    def to_dataset(self, number_of_frames=20, train_to_test_ratio=0.3):
        if not self.cache_exists():
            self.process_graph_sequences(number_of_frames)

            with open('dyngraph_embeddings.pkl', 'wb') as f:
                pkl.dump((self.scenegraphs_sequence, self.feature_list), f)
                
        else:
            with open(self.cache_filename,'rb') as f: 
                self.scenegraphs_sequence , self.feature_list = pkl.load(f)
            
        train, test = train_test_split(self.scenegraphs_sequence , test_size=train_to_test_ratio, shuffle=True)
        unzip_training_data = list(zip(*train)) 
        unzip_testing_data  = list(zip(*test))
        return_values = np.array(unzip_training_data[0]), np.array(unzip_training_data[1]), np.array(unzip_testing_data[0]), np.array(unzip_testing_data[1]), self.feature_list
        return return_values
        
    def get_node_embeddings(self, scenegraph):
        rows = []
        labels=[]
        ego_attrs = None
        
        #extract ego attrs for creating relative features
        for node, data in scenegraph.g.nodes.items():
            if "ego:" in str(node):
                ego_attrs = data['attr']
        if ego_attrs == None:
            raise NameError("Ego not found in scenegraph")
            
        def get_embedding(node, row):
            #subtract each vector from corresponding vector of ego to find delta
            if "location" in node.attr:
                row["rel_location_x"] = node.attr["location"][0] - ego_attrs["location"][0]
                row["rel_location_y"] = node.attr["location"][1] - ego_attrs["location"][1]
                row["rel_location_z"] = node.attr["location"][2] - ego_attrs["location"][2]
                row["distance_abs"] = math.sqrt(row["rel_location_x"]**2 + row["rel_location_y"]**2 + row["rel_location_z"]**2)
            row['type_'+str(node.type)] = 1 #assign 1hot class label
            return row
        
        for idx, node in enumerate(scenegraph.g.nodes):
            d = defaultdict()
            row = get_embedding(node, d)
            labels.append(node.type)
            rows.append(row)
            
        embedding = pd.DataFrame(data=rows, columns=self.feature_list)
        embedding = embedding.fillna(value=0) #fill in NaN with zeros
        embedding = torch.FloatTensor(embedding.values)
        
        return embedding

    def get_edge_embeddings(self, scenegraph, node_name2idx):
        edge_index = []
        edge_attr = []
        for src, dst, edge in scenegraph.g.edges(data=True):
            edge_index.append((node_name2idx[src], node_name2idx[dst]))
            edge_attr.append(edge['object'].value)

        edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
        edge_attr  = torch.LongTensor(edge_attr)
        
        return edge_index, edge_attr