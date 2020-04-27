import os, pdb, sys
sys.path.append(os.path.dirname(sys.path[0]))
from core.graph_learning import utils
from core.graph_learning.models import base_model
import pickle as pkl

from pygcn.models import GCN
from pygcn.utils import sparse_mx_to_torch_sparse_tensor, normalize, accuracy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp

from core.scene_graph.scene_graph import SceneGraphExtractor
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm




#loads data from processed_scenes folder. loads labels if they exist.
# def load_data(input_source):
#     scenegraphs = embeddings = adj_matrix = labels = None
    
#     if not input_source.exists():
#         raise FileNotFoundError("path does not exist: " + input_source)
        
#     else:
#         with open(str(input_source/"scenegraphs.pkl"), 'rb') as f:
#             scenegraphs = pkl.load(f)
#         with open(str(input_source/"embeddings.pkl"), 'rb') as f:
#             embeddings = pkl.load(f)
#         with open(str(input_source/"adj_matrix.pkl"), 'rb') as f:
#             adj_matrix = pkl.load(f)
#         if (input_source/"labels.pkl").exists():
#             with open(str(input_source/"labels.pkl"), 'rb') as f:
#                 labels = pkl.load(f)
                
#     return scenegraphs, embeddings, adj_matrix, labels

def reduce_scene_graph_extraction(args):
    args = SceneGraphArgumentParser().get_args(args)

    input_base_dir = Path(args.input_path).resolve()
    input_source   = input_base_dir / "processed_scenes"
    data_source    = input_base_dir / "scene_raw"

    # FIX ME 
    # save_dir       = input_base_dir / "models"
    
    # load scene graph txts into memory 
    sge = SceneGraphExtractor()
    sge.load(data_source)

    scene_graph_embeddings = {}
    scene_graph_labels = {}
    adj_matrix = {}

    feature_list = utils.get_feature_list(sge.scenegraphs.values(), num_classes=8)

    for timeframe, scenegraph in sge.scenegraphs.items():
        print(timeframe, scenegraph)

        scene_graph_labels[timeframe], scene_graph_embeddings[timeframe] = utils.create_node_embeddings(scenegraph, feature_list)
        adj_matrix[timeframe] = utils.get_adj_matrix(scenegraph)
    
    return sge.scenegraphs, adj_matrix, scene_graph_embeddings, scene_graph_labels

class SceneGraphArgumentParser:
    '''
        Argument Parser for script to train scenegraphs.
    '''
    def __init__(self):
        self.parser = ArgumentParser(description='The parameters for training the scene graph using GCN.')
        self.parser.add_argument('-ip', '--input_path', type=str, default="../input/synthesis_data/lane-change-9.8/", help="Path to code directory.")
        self.parser.add_argument('-lr',  dest='learning_rate', default=0.0001, type=float, help='The learning rate for GCN.')


    def get_args(self, args):
        return self.parser.parse_args(args)

class GCNTrainer:

    def __init__(self, args):
        self.scenegraphs, self.adj_matrixes, self.scene_graph_embeddings, self.scene_graph_labels = reduce_scene_graph_extraction(args)
        self.n_features = next(iter(self.scene_graph_embeddings.values())).shape[1]

    def build_model(self):
        #returns an embedding for each node (unsupervised)
        self.model = GCN(nfeat=self.n_features, nhid=200, nclass=8, dropout=0.75)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def train_model(self):

        features = list(self.scene_graph_embeddings.values())
        adjs =  list(self.adj_matrixes.values())
        labels =  list(self.scene_graph_labels.values())
        
        for epoch_idx in tqdm(range(100)): # iterate through epoch
            acc_loss_train = 0

            for i in range(len(features)): # iterate through scenegraphs
            
                self.model.train()
                self.optimizer.zero_grad()
                
                embs = torch.FloatTensor(np.array(normalize(sp.csr_matrix(features[i].values)).todense()))
                adj = sparse_mx_to_torch_sparse_tensor(normalize(adjs[i] + sp.eye(adjs[i].shape[0])))
               
                output = self.model.forward(embs, adj)

                loss_train = F.nll_loss(output, torch.LongTensor(labels[i]))

                loss_train.backward()
                self.optimizer.step()

                acc_loss_train += loss_train.item()

            print('Epoch: {:04d}'.format(epoch_idx), 'loss_train: {:.4f}'.format(acc_loss_train))

    def predict_node_classification(self):
        # take training set as testing data temporarily
        features = list(self.scene_graph_embeddings.values())
        adjs =  list(self.adj_matrixes.values())
        labels =  list(self.scene_graph_labels.values())

        for i in range(len(features)): # iterate through scenegraphs
    
            self.model.eval()
            
            embs = torch.FloatTensor(np.array(normalize(sp.csr_matrix(features[i].values)).todense()))
            adj = sparse_mx_to_torch_sparse_tensor(normalize(adjs[i] + sp.eye(adjs[i].shape[0])))
           
            output = self.model.forward(embs, adj)
            
            acc_train = accuracy(output, torch.LongTensor(labels[i]))

            print('SceneGraph: {:04d}'.format(i), 'acc_train: {:.4f}'.format(acc_train.item()))


if __name__ == "__main__":
    gcn_trainer = GCNTrainer(sys.argv[1:])
    gcn_trainer.build_model()
    gcn_trainer.train_model()
    gcn_trainer.predict_node_classification()