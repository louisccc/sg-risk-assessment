import os, pdb, sys
sys.path.append(os.path.dirname(sys.path[0]))
from core.graph_learning import utils
from core.graph_learning.bases import BaseTrainer

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import pandas as pd

from core.scene_graph.graph_process import NodeClassificationExtractor
from core.graph_learning.models.gcn import *
from core.graph_learning.models.gin import *
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader

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
        self.parser.add_argument('--device', type=str, default="cpu", help='The device to run on models (cuda or cpu) cpu in default.')
        self.parser.add_argument('--model', type=str, default="gcn", help="Model to be used intrinsically.")

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()


class GCNTrainer(BaseTrainer):

    def __init__(self, args):
        self.config = Config(args)

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        self.preprocess_scenegraph_data() # reduced scenegraph extraction

    def preprocess_scenegraph_data(self):
        # load scene graph txts into memory 
        sge = NodeClassificationExtractor()

        if not sge.is_cache_exists():
            if self.config.recursive:
                for sub_dir in tqdm([x for x in self.config.input_base_dir.iterdir() if x.is_dir()]):
                    sge.load(sub_dir)
            else:
                sge.load(self.config.input_base_dir)

            self.train_graphs, self.test_graphs = sge.to_dataset(train_to_test_ratio=0.1)

        else:
            self.train_graphs, self.test_graphs = sge.read_cache()

        self.num_features = self.train_graphs[0].node_features.shape[1]
        self.num_training_samples = len(self.train_graphs)
        self.num_testing_samples  = len(self.test_graphs)
        
        # data loader for training and testing
        train_data_list = [Data(x=g.node_features, edge_index=g.edge_mat, y=torch.LongTensor(g.node_labels)) for g in self.train_graphs]
        self.train_loader = DataLoader(train_data_list, batch_size=32)
        test_data_list = [Data(x=g.node_features, edge_index=g.edge_mat, y=torch.LongTensor(g.node_labels)) for g in self.test_graphs]
        self.test_loader = DataLoader(test_data_list, batch_size=1)

        print("Number of SceneGraphs in the training set: ", self.num_training_samples)
        print("Number of SceneGraphs in the testing set:  ", self.num_testing_samples)
        print("Number of nodes in the training set:", sum([g.node_features.shape[0] for g in self.train_graphs]))
        print("Number of nodes in the testing set: ", sum([g.node_features.shape[0] for g in self.test_graphs]))
        print("Number of features for each node: ", self.num_features)


    def build_model(self):
        if self.config.model == "gcn":
            self.model = GCN(self.num_features, self.config.hidden, self.config.nclass, self.config.dropout).to(self.config.device)
        elif self.config.model == "gin":
            self.model = GIN(None, self.num_features, 2).to(self.config.device)
                
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def train(self):

        data = self.train_graphs

        for epoch_idx in tqdm(range(self.config.epochs)):
            acc_loss_train = 0

            for data in self.train_loader: 
                
                data.to(self.config.device)

                self.model.train()
                self.optimizer.zero_grad()

                output = self.model.forward(data.x, data.edge_index)
                
                loss_train = nn.CrossEntropyLoss()(output, data.y)

                loss_train.backward()
                self.optimizer.step()

                acc_loss_train += loss_train.item()

            print('Epoch: {:04d}'.format(epoch_idx), 'loss_train: {:.4f}'.format(acc_loss_train))

    def predict(self):
        # predict the node classification performance.
        outputs = []
        labels = []
        
        for i, data in enumerate(self.test_loader):
            
            data.to(self.config.device)

            self.model.eval()
                     
            output = self.model.forward(data.x, data.edge_index)
            
            acc_test = utils.accuracy(output, data.y)

            print('SceneGraph: {:04d}'.format(i), 'acc_test: {:.4f}'.format(acc_test.item()))
            
            outputs.append(output.cpu())
            labels.append(data.y.cpu().numpy())
        
        return torch.cat(outputs).detach(), np.concatenate(labels)
        
        