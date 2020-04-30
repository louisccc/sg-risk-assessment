import os, pdb, sys
sys.path.append(os.path.dirname(sys.path[0]))
from core.graph_learning import utils
from core.graph_learning.models import base_model

from pygcn.models import GCN
from pygcn.utils import sparse_mx_to_torch_sparse_tensor, normalize, accuracy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import pandas as pd

from core.scene_graph.graph_process import NodeClassificationExtractor
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


class Config:

    '''Argument Parser for script to train scenegraphs.'''

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for training the scene graph using GCN.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data/lane-change/0", help="Path to code directory.")
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='The initial learning rate for GCN.')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        self.parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--hidden', type=int, default=200, help='Number of hidden units.')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--nclass', type=int, default=8, help="The number of classes for node.")
        self.parser.add_argument('--recursive', type=lambda x: (str(x).lower() == 'true'), default=False, help='Recursive loading scenegraphs')

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()


class GCNTrainer:

    def __init__(self, args):
        self.config = Config(args)

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        self.preprocess_scenegraph_data() # reduced scenegraph extraction

    def preprocess_scenegraph_data(self):
        # load scene graph txts into memory 
        sge = NodeClassificationExtractor()

        if self.config.recursive:
            for sub_dir in tqdm([x for x in self.config.input_base_dir.iterdir() if x.is_dir()]):
                sge.load(sub_dir)
        else:
            sge.load(self.config.input_base_dir)

        self.node_embeddings = []
        self.node_labels = []
        self.adj_matrixes = []

        self.training_data, self.testing_data = sge.to_dataset()
        
        unzip_training_data = list(zip(*self.training_data)) 
        unzip_testing_data = list(zip(*self.testing_data))

        self.node_embeddings, self.node_labels, self.adj_matrixes = list(unzip_training_data[0]), list(unzip_training_data[1]), list(unzip_training_data[2])
        self.node_embeddings_test, self.node_labels_test, self.adj_matrixes_test = list(unzip_testing_data[0]), list(unzip_testing_data[1]), list(unzip_testing_data[2])                    

        self.n_features = self.node_embeddings[0].shape[1]

        print("Number of Scene Graphs included: ", len(self.node_embeddings))

    def build_model(self):
        #returns an embedding for each node (unsupervised)
        self.model = GCN(nfeat=self.n_features, nhid=self.config.hidden, nclass=self.config.nclass, dropout=self.config.dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def train_model(self):

        features = self.node_embeddings
        adjs =  self.adj_matrixes
        labels =  self.node_labels
        
        for epoch_idx in tqdm(range(self.config.epochs)): # iterate through epoch
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
        features = self.node_embeddings_test
        adjs =  self.adj_matrixes_test
        labels =  self.node_labels_test

        result_embeddings = pd.DataFrame()
        for i in range(len(features)): # iterate through scenegraphs
    
            self.model.eval()
            
            embs = torch.FloatTensor(np.array(normalize(sp.csr_matrix(features[i].values)).todense()))
            adj = sparse_mx_to_torch_sparse_tensor(normalize(adjs[i] + sp.eye(adjs[i].shape[0])))
           
            output = self.model.forward(embs, adj)
            result_embeddings = pd.concat([result_embeddings, pd.DataFrame(output.detach().numpy())], axis=0, ignore_index=True)
            acc_train = accuracy(output, torch.LongTensor(labels[i]))

            print('SceneGraph: {:04d}'.format(i), 'acc_train: {:.4f}'.format(acc_train.item()))
        
        self.save_output(np.concatenate(labels), result_embeddings, "test")
    
    #generate TSV output file from embeddings and labels for visualization
    def save_output(self, metadata, embeddings, filename):
        pd.DataFrame(metadata).to_csv(self.config.input_base_dir / str(filename + "_meta.tsv"), sep='\t', header=False, index=False)
        embeddings.to_csv(self.config.input_base_dir / str(filename + "_embeddings.tsv"), sep="\t", header=False, index=False)


if __name__ == "__main__":
    gcn_trainer = GCNTrainer(sys.argv[1:])
    gcn_trainer.build_model()
    gcn_trainer.train_model()
    gcn_trainer.predict_node_classification()