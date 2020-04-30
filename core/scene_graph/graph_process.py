import numpy as np
import networkx as nx
import pickle as pkl
import pandas as pd

from .scene_graph import SceneGraph
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split

from pygcn.utils import sparse_mx_to_torch_sparse_tensor, normalize, accuracy
import scipy.sparse as sp

import torch, json


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

    def to_dataset(self, train_to_test_ratio=0.1):
        
        node_embeddings = []
        node_labels = []
        adj_matrixes = []

        feature_list = self.get_feature_list(num_classes=8)

        for scenegraphs in self.scenegraphs_sequence:
            for timeframe, scenegraph in scenegraphs.items():
                labels, embeddings = self.create_node_embeddings(scenegraph, feature_list)
                adjs = self.get_adj_matrix(scenegraph)
                
                embeddings = torch.FloatTensor(np.array(normalize(sp.csr_matrix(embeddings.values)).todense()))
                adjs = sparse_mx_to_torch_sparse_tensor(normalize(adjs + sp.eye(adjs.shape[0])))

                node_embeddings.append(embeddings)
                node_labels.append(labels)
                adj_matrixes.append(adjs)

        # training, testing set split.
        train, test = train_test_split(list(zip(node_embeddings, node_labels, adj_matrixes)), test_size=train_to_test_ratio, shuffle=True)
        
        # in train and test, each row stands for a scenegraph: 
        # 1) a list of node embeddings
        # 2) a list of node labels
        # 3) adjacency matrix for this scenegraph
        # return node_embeddings, node_labels, adj_matrixes

        return train, test
    
    #gets a list of all feature labels for all scenegraphs
    def get_feature_list(self, num_classes):
        all_attrs = set()
        for scenegraphs in self.scenegraphs_sequence:
            for timeframe, scenegraph in scenegraphs.items():
                for entity in scenegraph.g.nodes:
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
        for node in scenegraph.g.nodes:
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
        
        embedding = pd.DataFrame(data=rows, columns=feature_list)
        embedding = embedding.fillna(value=0) #fill in NaN with zeros
        
        return np.array(labels), embedding

    #get adjacency matrix for entity nodes only from  scenegraph in scipy.sparse CSR matrix format
    def get_adj_matrix(self, scenegraph):
        adj = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g)
        return adj


class SceneGraphExtractor(NodeClassificationExtractor):
    
    def __init__(self):      
        self.scenegraphs_sequence = []

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

        print(path, risk_label)
        self.scenegraphs_sequence.append((scenegraphs, risk_label))
    
    #gets a list of all feature labels for all scenegraphs
    def get_feature_list(self, num_classes):
        all_attrs = set()
        for scenegraphs, risk_label in self.scenegraphs_sequence:
            for timeframe, scenegraph in scenegraphs.items():
                for entity in scenegraph.g.nodes:
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
        for node in scenegraph.g.nodes:
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
        adj = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g)
        return adj

    def create_dataset_4_graph_classification(self):
        graph_labels = []
        graphs = []

        feature_list = self.get_feature_list(num_classes=8)

        for scenegraphs, risk_label in self.scenegraphs_sequence:
            for timeframe, scenegraph in scenegraphs.items():
                graphs.append(scenegraph)
                graph_labels.append(risk_label)
                _, node_features = self.create_node_embeddings(scenegraph, feature_list)
                scenegraph.node_features = torch.FloatTensor(node_features.values)
                
                sparse_mx = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g).tocoo().astype(np.float32)
                scenegraph.edge_mat = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

        return graphs, graph_labels


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