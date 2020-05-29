import os, sys, pdb
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import pandas as pd

from core.graph_learning.bases import BaseTrainer
from core.scene_graph.graph_process import SceneGraphSequenceGenerator
from core.graph_learning.utils import accuracy
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from core.graph_learning.models.gcn import *
from core.graph_learning.models.gin import *
from torch_geometric.data import Data, DataLoader

class Config:
    '''Argument Parser for script to train scenegraphs.'''
    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for training the scene graph using GCN.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data/lane-change/", help="Path to code directory.")
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='The initial learning rate for GCN.')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        self.parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--hidden', type=int, default=200, help='Number of hidden units.')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--nclass', type=int, default=8, help="The number of classes for node.")
        self.parser.add_argument('--recursive', type=lambda x: (str(x).lower() == 'true'), default=True, help='Recursive loading scenegraphs')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Number of graphs in a batch.')
        self.parser.add_argument('--device', type=str, default="cpu", help='The device to run on models (cuda or cpu) cpu in default.')
        self.parser.add_argument('--test_step', type=int, default=10, help='Number of epochs before testing the model.')
        self.parser.add_argument('--model', type=str, default="gcn", help="Model to be used intrinsically.")
        
        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()


class DynGraphTrainer(BaseTrainer):

    def __init__(self, args):
        self.config = Config(args)

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        self.preprocess_scenegraph_data() # reduced scenegraph extraction

    def preprocess_scenegraph_data(self):
        # load scene graph txts into memory 
        sge = SceneGraphSequenceGenerator()

        if not sge.is_cache_exists():
            if self.config.recursive:
                for sub_dir in tqdm([x for x in self.config.input_base_dir.iterdir() if x.is_dir()]):
                    data_source = sub_dir
                    sge.load(data_source)
            else:
                data_source = self.config.input_base_dir
                sge.load(data_source)

            self.training_sequences, self.training_labels, self.testing_sequences, self.testing_labels, self.feature_list = sge.to_dataset()
        
        else:
            self.training_sequences, self.training_labels, self.testing_sequences, self.testing_labels, self.feature_list = sge.read_cache()

        print("Number of Sequences included: ", len(self.training_sequences))


    def build_model(self):
        if self.config.model == "gcn":
            self.model = GCN_Graph_Sequence(len(self.feature_list), self.config.hidden, 2, 0.75, "mean", "mean").to(self.config.device)
        elif self.config.model == "gin":
            self.model = GIN_Graph_Sequence(None, len(self.feature_list), 2, "mean").to(self.config.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def train(self):

        for epoch_idx in tqdm(range(self.config.epochs)): # iterate through epoch
            acc_loss_train = 0
            for i in range(len(self.training_sequences)): # iterate through scenegraphs
                data, label = self.training_sequences[i], self.training_labels[i]

                data_list = [Data(x=g.node_features, edge_index=g.edge_mat) for g in data]
                self.train_loader = DataLoader(data_list, batch_size=len(data_list))
                sequence = next(iter(self.train_loader)).to(self.config.device)

                self.model.train()
                self.optimizer.zero_grad()
                               
                output = self.model.forward(sequence.x, sequence.edge_index, sequence.batch)
                
                loss_train = nn.CrossEntropyLoss()(output.view(-1, 2), torch.LongTensor([label]).to(self.config.device))

                loss_train.backward()

                self.optimizer.step()

                acc_loss_train += loss_train.detach().cpu().numpy()

            print('')
            print('Epoch: {:04d},'.format(epoch_idx), 'loss_train: {:.4f}'.format(acc_loss_train))
            print('')

            # if epoch_idx % self.config.test_step == 0:
            #     self.predict()

    def predict(self):
        # take training set as testing data temporarily
        acc_predict = []
        labels = []
        outputs = []
        
        for i in range(len(self.testing_sequences)): # iterate through scenegraphs
            data, label = self.testing_sequences[i], self.testing_labels[i]
            
            self.model.eval()
            output = self.model.forward(data)
            outputs.append(output)
            labels.append(label)
            
            print(output, label)
            acc_test = accuracy(output.view(-1, 2), torch.LongTensor([label]).to(self.config.device))
            acc_predict.append(acc_test.item())

            print('Dynamic SceneGraph: {:04d}'.format(i), 'acc_test: {:.4f}'.format(acc_test.item()))

        print('Dynamic SceneGraph precision', sum(acc_predict) / len(acc_predict))
        outputs = torch.cat(outputs).reshape(-1,2).detach()
        if self.config.device == "cuda":
            outputs = outputs.cpu()
        return outputs, np.array(labels).flatten()