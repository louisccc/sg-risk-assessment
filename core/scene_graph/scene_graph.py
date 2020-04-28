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
    def __init__(self, framedict=None):
        self.g = nx.Graph() #initialize scenegraph as networkx graph
        self.relation_extractor = RelationExtractor()
        self.road_node = Node("road", {}, ActorType.ROAD, True)
        self.entity_nodes = []  #nodes which are explicit entities and not attributes.
        self.add_node(self.road_node)   #adding the road as the root node
        if framedict != None:
            self.add_frame_dict(framedict)
        
    #add single node to graph. node can be any hashable datatype including objects.
    def add_node(self, node):
        self.g.add_node(node, attr=node.attr, label=node.label)
        if(node.is_entity):
            self.entity_nodes.append(node)
        
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
        for actor_id, attr in actordict.items():
            n = Node(actor_id, attr, None, True)   #using the actor key as the node name and the dict as its attributes.
            n.type = self.relation_extractor.get_actor_type(n).value
            self.add_node(n)
            self.add_attributes(n, attr)
            
    #adds lanes and their dicts. constructs relation between each lane and the root road node.
    def add_lane_dict(self, lanedict):
        
        n = Node(str(lanedict['ego_lane']['lane_id']), lanedict['ego_lane'], ActorType.LANE, True) #todo: change to true when lanedict entry is complete
        self.add_node(n)
        self.add_relation([n, Relations.partOf, self.road_node])

        for lane in lanedict['left_lanes']:
            # for lane_id, laneattr in lane.items():
            n = Node(str(lane['lane_id']), lane, ActorType.LANE, True) #todo: change to true when lanedict entry is complete
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])

        for lane in lanedict['right_lanes']:
            n = Node(str(lane['lane_id']), lane, ActorType.LANE, True) #todo: change to true when lanedict entry is complete
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])
            
    #add signs as entities of the road.
    def add_sign_dict(self, signdict):
        for sign_id, signattr in signdict.items():
            n = Node(sign_id, signattr, ActorType.SIGN, True)
            self.add_node(n)
            self.add_relation([n, Relations.partOf, self.road_node])

    #parses attributes of ego/actors
    def add_attributes(self, node, attrdict):
        for attr, values in attrdict.items():
            n = Node(attr, values, None, False)   #attribute nodes are not entities themselves
            self.add_node(n)
            self.add_relation([node, Relations.hasAttribute, n])

    #add the contents of a whole framedict to the graph
    def add_frame_dict(self, framedict):
        for key, attrs in framedict.items():
            # import pdb; pdb.set_trace()
            if key == "ego":
                egoNode = Node(key, attrs, ActorType.CAR, True)
                self.add_node(egoNode)
                self.add_attributes(egoNode, attrs)
            elif key == "lane":
                self.add_lane_dict(attrs)
            elif key == "sign":
                self.add_sign_dict(attrs)
            else:
                self.add_actor_dict(attrs)

            
    #calls RelationExtractor to build semantic relations between every pair of entity nodes in graph. call this function after all nodes have been added to graph.
    def extract_semantic_relations(self):
        for node1 in self.entity_nodes:
            for node2 in self.entity_nodes:
                if node1.name != node2.name and node1.name != 'road' and node2.name != 'road': #dont build self-relations or relations w/ road
                    self.add_relations(self.relation_extractor.extract_relations(node1, node2))
    

class SceneGraphExtractor:
    def __init__(self):
        self.scenegraphs_sequence = []

    def load(self, path):
        scenegraphs = {}

        for txt_path in glob("%s/**/*.txt" % str(path), recursive=True):
            scene_dict_f = open(txt_path, 'r')
            
            framedict = json.loads(scene_dict_f.read())

            for frame, frame_dict in framedict.items():
                scenegraph = SceneGraph()
                scenegraph.add_frame_dict(frame_dict)
                scenegraph.extract_semantic_relations()
                scenegraphs[frame] = scenegraph
        
        risk_label = 1.0

        self.scenegraphs_sequence.append((scenegraphs, risk_label))

    #gets a list of all feature labels for all scenegraphs
    def get_feature_list(self, num_classes):
        all_attrs = set()
        for scenegraphs, risk_label in self.scenegraphs_sequence:
            for timeframe, scenegraph in scenegraphs.items():
                for entity in scenegraph.entity_nodes:
                    all_attrs.update(entity.attr.keys())
                    
        final_attr_list = all_attrs.copy()
        for attr in all_attrs:
            if attr in ["location", "rotation", "velocity", "ang_velocity"]:
                final_attr_list.discard(attr)
                final_attr_list.update([attr+"_x", attr+"_y", attr+"_z"]) #add 3 columns for vector values
        
        for i in range(num_classes):
            final_attr_list.add("type_"+str(i)) #create 1hot class labels
        
        final_attr_list.discard("name") #remove node name as it is not needed sice we have class labels
        
        return sorted(final_attr_list)

    def create_node_embeddings(self, scenegraph, feature_list):
        rows = []
        labels=[]
        for node in scenegraph.entity_nodes:
            row = defaultdict()
            for attr in node.attr:
                if attr in ["location", "rotation", "velocity", "ang_velocity"]:
                    row[attr+"_x"] = node.attr[attr][0]
                    row[attr+"_y"] = node.attr[attr][1]
                    row[attr+"_z"] = node.attr[attr][2]
                elif attr == "is_junction": #binarizing junction label
                    row[attr] = 1 if node.attr==True else 0
                elif attr == "name": #dont add name to embedding
                    continue
                else:
                    row[attr] = node.attr[attr]
            row['type_'+str(node.type)] = 1 #assign 1hot class label
            labels.append(node.type)
            rows.append(row)
        #pdb.set_trace()
        embedding = pd.DataFrame(data=rows, columns=feature_list)
        embedding = embedding.fillna(value=0) #fill in NaN with zeros
        
        return np.array(labels), embedding

    #get adjacency matrix for entity nodes only from  scenegraph in scipy.sparse CSR matrix format
    def get_adj_matrix(self, scenegraph):
        adj = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g, nodelist=scenegraph.entity_nodes)
        return adj

    def create_dataset_4_node_classification(self):
        node_embeddings = []
        node_labels = []
        adj_matrixes = []

        feature_list = self.get_feature_list(num_classes=8)
        for scenegraphs, risk_label in self.scenegraphs_sequence:
            for timeframe, scenegraph in scenegraphs.items():
                labels, embeddings = self.create_node_embeddings(scenegraph, feature_list)
                adjs = self.get_adj_matrix(scenegraph)
                node_embeddings.append(embeddings)
                node_labels.append(labels)
                adj_matrixes.append(adjs)

        # training, testing set split.

        # each row stands for a scenegraph: 
        # 1) a list of node embeddings
        # 2) a list of node labels
        # 3) adjacency matrix for this scenegraph
        return node_embeddings, node_labels, adj_matrixes

    # self.scene_images = {}
    # For Visualization
    # self.fig, (self.ax_graph, self.ax_img) = plt.subplots(1, 2, figsize=(20, 12))
    # self.fig.canvas.set_window_title("Scene Graph Visualization")

    # TODO: Aung change this into gif creations.
    # def build_corresponding_images(self, path):
    #     for scenegraphs, risk_label in self.scenegraphs_sequence.items():
    #         for frame, scenegraph in scenegraphs.items():
    #             try:
    #                 print('catch %s/%.8d.png'%(path, int(frame)))
    #                 img = plt.imread('%s/%.8d.png'%(path, int(frame)))
    #                 self.scene_images[frame] = (scenegraph, img)
    #             except Exception as e:
    #                 print(e)
        
    # def store(self, path):
    #     for frame, (scenegraph, image) in self.scene_images.items():
    #         pos = nx.spring_layout(scenegraph.g, k=1.5*1/np.sqrt(len(scenegraph.g.nodes())))
    #         nx.draw(scenegraph.g, labels=nx.get_node_attributes(scenegraph.g, 'label'), pos=pos, font_size=8, with_labels=True, ax=self.ax_graph)
    #         self.ax_img.imshow(image)
    #         self.ax_graph.set_title("Risk {}".format(random.random()))
    #         self.ax_img.set_title("Frame {}".format(frame))
    #         plt.savefig('%s/%s.png'%(path, frame))
    #         self.ax_graph.clear()
    #         self.ax_img.clear()

    # def update(self, num):
    #     self.ax_graph.clear()
    #     self.ax_img.clear()

    #     frame = list(self.scene_images.keys())[num]
    #     scenegraph = self.scene_images[frame][0]
    #     pos = nx.spring_layout(scenegraph.g, k=1.5*1/np.sqrt(len(scenegraph.g.nodes())))
    #     nx.draw(scenegraph.g, labels=nx.get_node_attributes(scenegraph.g, 'label'), pos=pos, font_size=8, with_labels=True, ax=self.ax_graph)
    #     self.ax_img.imshow(self.scene_images[frame][1])

    #     # Set the title
    #     self.ax_graph.set_title("Risk {}".format(random.random()))
    #     self.ax_img.set_title("Frame {}".format(num))

    # def show_animation(self):
    #     ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.scene_images.keys()))
    #     plt.show()