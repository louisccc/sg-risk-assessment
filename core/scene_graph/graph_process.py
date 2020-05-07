import numpy as np
import networkx as nx
import pickle as pkl
import pandas as pd
import math

from .scene_graph import SceneGraph
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split

from pygcn.utils import sparse_mx_to_torch_sparse_tensor, normalize, accuracy
import scipy.sparse as sp

import torch, json, pdb
import pickle as pkl
from pathlib import Path 


LANE_MARKING_TYPES = [
"NONE",
"Other",
"Broken",
"Solid",
"SolidSolid",
"SolidBroken",
"BrokenSolid",
"BrokenBroken",
"BottsDots",
"Grass",
"Curb"]

LANE_TYPES = [
"NONE",
"Driving",
"Stop",
"Shoulder",
"Biking",
"Sidewalk",
"Border",
"Restricted",
"Parking",
"Bidirectional",
"Median",
"Special1",
"Special2",
"Special3",
"RoadWorks",
"Tram",
"Rail",
"Entry",
"Exit",
"OffRamp",
"OnRamp",
"Any"]

LANE_MARKING_COLORS = [
"Standard"
"Blue"
"Green"
"Red"
"White"
"Yellow"
"Other"]


class NodeClassificationExtractor: 
    
    def __init__(self):
        # each item in self.scenegraphs_sequence is a dictionary of scenegraphs.
        self.scenegraphs_sequence = []

    def load(self, path):
        scenegraphs = {}

        for txt_path in glob("%s/**/*.txt" % str(path/"scene_raw"), recursive=True):
            scene_dict_f = open(txt_path, 'r')
            
            framedict = json.loads(scene_dict_f.read())

            for frame, frame_dict in framedict.items():
                scenegraph = SceneGraph(frame_dict)
                scenegraphs[frame] = scenegraph

        self.scenegraphs_sequence.append(scenegraphs)

    def is_cache_exists(self):
        return Path('node_embeddings.pkl').exists()

    def read_cache(self):   
        with open('node_embeddings.pkl','rb') as f: 
            return pkl.load(f)

    def to_dataset(self, train_to_test_ratio=0.3):
        
        graphs = []
        feature_list = self.get_feature_list(num_classes=8)

        for scenegraphs in self.scenegraphs_sequence:
            for timeframe, scenegraph in scenegraphs.items():
                labels, embeddings = self.create_node_embeddings(scenegraph, feature_list)
                adjs = self.get_adj_matrix(scenegraph)
                
                embeddings = torch.FloatTensor(np.array(normalize(sp.csr_matrix(embeddings.values)).todense()))
                adjs = sparse_mx_to_torch_sparse_tensor(normalize(adjs + sp.eye(adjs.shape[0])))

                scenegraph.node_features = embeddings
                scenegraph.node_labels = labels
                scenegraph.adj_matrix = adjs
                
                sparse_mx = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g).tocoo().astype(np.float32)
                scenegraph.edge_mat = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

                graphs.append(scenegraph)

        train, test = train_test_split(graphs, test_size=train_to_test_ratio, shuffle=True)
        
        # in train and test, each row stands for a scenegraph: 
        # 1) a list of node embeddings
        # 2) a list of node labels
        # 3) adjacency matrix for this scenegraph
        # return node_embeddings, node_labels, adj_matrixes

        return_values = train, test

        # Saving the objects:
        with open('node_embeddings.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pkl.dump(return_values, f)

        return return_values
    
    #gets a list of all feature labels (which will be used) for all scenegraphs
    def get_feature_list(self, num_classes):
        all_attrs = set()
        
        def get_all_attrs(scenegraphs, attrs):
            for timeframe, scenegraph in scenegraphs.items():
                for entity in scenegraph.g.nodes:
                    attrs.update(entity.attr.keys())
            return attrs
        
        for scenegraphs in self.scenegraphs_sequence:
            if type(scenegraphs) is tuple:
                scenegraphs = scenegraphs[0]
            all_attrs = get_all_attrs(scenegraphs, all_attrs)
                    
        final_attr_list = all_attrs.copy()
        for attr in all_attrs:
            if attr in ["location", "velocity", "ang_velocity", "rotation"]:
                final_attr_list.discard(attr)
                final_attr_list.update(["rel_"+attr+"_x", "rel_"+attr+"_y", "rel_"+attr+"_z"]) #add 3 columns for relative vector values
        final_attr_list.add("distance_abs") #adding absolute distance to ego
        
        for i in range(num_classes):
            final_attr_list.add("type_"+str(i)) #create 1hot class labels
        for i in range(len(LANE_TYPES)):
            final_attr_list.add("lane_type_"+str(i))
        for i in range(len(LANE_MARKING_COLORS)):
            final_attr_list.add("left_lane_color_"+str(i))
            final_attr_list.add("right_lane_color_"+str(i))
        for i in range(len(LANE_MARKING_TYPES)):
            final_attr_list.add("left_lane_marking_type_"+str(i))
            final_attr_list.add("right_lane_marking_type_"+str(i))
            
        final_attr_list.discard("name") #remove node name as it is not needed sice we have class labels
        final_attr_list.discard("road_id") #remove road_id from feature list
        final_attr_list.discard("lane_id")
        
        return sorted(final_attr_list)

    def create_node_embeddings(self, scenegraph, feature_list):
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
            for attr in node.attr:
                if attr in ["location", "velocity", "ang_velocity", 'rotation']: #subtract each vector from corresponding vector of ego to find delta
                    row["rel_"+attr+"_x"] = node.attr[attr][0] - ego_attrs[attr][0]
                    row["rel_"+attr+"_y"] = node.attr[attr][1] - ego_attrs[attr][1]
                    row["rel_"+attr+"_z"] = node.attr[attr][2] - ego_attrs[attr][2]
                    if attr == 'location':
                        row["distance_abs"] = math.sqrt(row["rel_"+attr+"_x"]**2 + row["rel_"+attr+"_y"]**2 + row["rel_"+attr+"_z"]**2)
                elif attr == "is_junction": #binarizing junction label
                    row[attr] = 1 if node.attr[attr]==True else 0
                elif attr in ['brake_light_on', 'left_blinker_on', 'right_blinker_on']:
                    row[attr] = 1 if node.attr[attr] > 0 else 0 #binarize light signals
                elif attr in ['lane_type', 'left_lane_marking_type', 'right_lane_marking_type', 'left_lane_color', 'right_lane_color']:
                    row[attr+"_"+str(node.attr[attr])] = 1 #assign 1hot labels for lanes
                elif attr in feature_list: #only add attributes specified by feature_list
                    row[attr] = node.attr[attr]
            row['type_'+str(node.type)] = 1 #assign 1hot class label
            return row
        
        for node in scenegraph.g.nodes:
            d = defaultdict()
            row = get_embedding(node, d)
            labels.append(node.type)
            rows.append(row)
            
        embedding = pd.DataFrame(data=rows, columns=feature_list)
        embedding = embedding.fillna(value=0) #fill in NaN with zeros
        

        return np.array(labels), embedding

    #get adjacency matrix for entity nodes only from  scenegraph in scipy.sparse CSR matrix format
    def get_adj_matrix(self, scenegraph):
        adj = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g)
        return adj


class SceneGraphExtractor(NodeClassificationExtractor):
    
    def __init__(self):
        super(SceneGraphExtractor, self).__init__()

    def load(self, path):
        scenegraphs = {}

        for txt_path in glob("%s/**/*.txt" % str(path/"scene_raw"), recursive=True):
            scene_dict_f = open(txt_path, 'r')
            
            framedict = json.loads(scene_dict_f.read())

            for frame, frame_dict in framedict.items():
                scenegraph = SceneGraph(frame_dict)
                scenegraphs[frame] = scenegraph

        label_f = open(str(path/"label.txt"), 'r')
        risk_label = int(label_f.read())

        if risk_label >= 0:
            risk_label = 1
        else:
            risk_label = 0

        self.scenegraphs_sequence.append((scenegraphs, risk_label))
    
    def is_cache_exists(self):
        return Path('graph_embeddings.pkl').exists()

    def read_cache(self):   
        with open('graph_embeddings.pkl','rb') as f: 
            return pkl.load(f)

    def process_graph_sequence(self, feature_list, sequence):
        graph_labels = []
        graphs = []
        for scenegraphs, risk_label in sequence:
            for timeframe, scenegraph in scenegraphs.items():
                graphs.append(scenegraph)
                graph_labels.append(risk_label)
                _, node_features = self.create_node_embeddings(scenegraph, feature_list)
                scenegraph.node_features = torch.FloatTensor(node_features.values)
                
                sparse_mx = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g).tocoo().astype(np.float32)
                scenegraph.edge_mat = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        return list(zip(graphs, graph_labels))

    def to_dataset(self, train_to_test_ratio=0.3):
        feature_list = self.get_feature_list(num_classes=8)
        
        train_sequence, test_sequence = train_test_split(self.scenegraphs_sequence, test_size = train_to_test_ratio)
        train = self.process_graph_sequence(feature_list, train_sequence)
        test = self.process_graph_sequence(feature_list, test_sequence)

        unzip_training_data = list(zip(*train)) 
        unzip_testing_data  = list(zip(*test))

        # train_graphs, train_labels.
        # test_graphs, test_labels.

        return_values = np.array(unzip_training_data[0]), np.array(unzip_training_data[1]), np.array(unzip_testing_data[0]), np.array(unzip_testing_data[1]), feature_list

        # Saving the objects:
        with open('graph_embeddings.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pkl.dump(return_values, f)

        return return_values


class SceneGraphSequenceGenerator(SceneGraphExtractor):
    def __init__(self):
        super(SceneGraphSequenceGenerator, self).__init__()

    def is_cache_exists(self):
        return Path('dyngraph_embeddings.pkl').exists()

    def read_cache(self):   
        with open('dyngraph_embeddings.pkl','rb') as f: 
            return pkl.load(f)

    def to_dataset(self, number_of_frames=20, train_to_test_ratio=0.3):
        sequence_labels = []
        sequences = [] 

        feature_list = self.get_feature_list(num_classes=8)
        for scenegraphs, risk_label in self.scenegraphs_sequence:
            sequence = []
            acc_number = 0
            modulo = int(len(scenegraphs) / number_of_frames)
            for idx, (timeframe, scenegraph) in enumerate(scenegraphs.items()):
                if idx % modulo == 0 and acc_number < number_of_frames:
                    sequence.append(scenegraph)
                    
                    _, node_features = self.create_node_embeddings(scenegraph, feature_list)
                    scenegraph.node_features = torch.FloatTensor(node_features.values)
                    
                    sparse_mx = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g).tocoo().astype(np.float32)
                    scenegraph.edge_mat = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
                    acc_number+=1
            sequences.append(sequence)
            sequence_labels.append(risk_label)

        train, test = train_test_split(list(zip(sequences, sequence_labels)), test_size=train_to_test_ratio, shuffle=True)

        unzip_training_data = list(zip(*train)) 
        unzip_testing_data  = list(zip(*test))

        # train_sequences, train_sequence_labels.
        # test_sequences, test_sequence_labels.

        return_values = np.array(unzip_training_data[0]), np.array(unzip_training_data[1]), np.array(unzip_testing_data[0]), np.array(unzip_testing_data[1]), feature_list

        # Saving the objects:
        with open('dyngraph_embeddings.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pkl.dump(return_values, f)

        return return_values