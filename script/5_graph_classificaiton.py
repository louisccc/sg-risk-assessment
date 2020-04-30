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

from core.scene_graph.graph_process import SceneGraphExtractor
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from core.graph_learning.models.gin import *


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
        self.parser.add_argument('--batch_size', type=int, default=32, help='Number of graphs in a batch.')

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()


class Generator:

    def __init__(self, data, label, batch_size):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        
        self.number_of_batch = len(data) // self.batch_size
        
        self.random_ids = np.random.permutation(len(data))
        self.batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        pos_start = self.batch_size * self.batch_idx
        pos_end   = self.batch_size * (self.batch_idx+1)

        raw_data  = [self.data[x]  for x in self.random_ids[pos_start:pos_end]]
        raw_label = [self.label[x] for x in self.random_ids[pos_start:pos_end]]
        
        self.batch_idx += 1
        if  self.batch_idx == self.number_of_batch:
            self.batch_idx = 0
            self.random_ids = np.random.permutation(len(self.data))

        return raw_data, raw_label

class GINTrainer:

    def __init__(self, args):
        self.config = Config(args)

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        self.preprocess_scenegraph_data() # reduced scenegraph extraction

    def preprocess_scenegraph_data(self):
        # load scene graph txts into memory 
        sge = SceneGraphExtractor()

        if self.config.recursive:
            for sub_dir in tqdm([x for x in self.config.input_base_dir.iterdir() if x.is_dir()]):
                data_source = sub_dir
                sge.load(data_source)
        else:
            data_source = self.config.input_base_dir
            sge.load(data_source)

        self.training_graphs, self.training_labels = sge.to_dataset()
        
        self.generator = Generator(self.training_graphs, self.training_labels, self.config.batch_size)

        print("Number of Scene Graphs included: ", len(self.training_graphs))

    def build_model(self):
        self.model = GraphCNN(4, 4, 34, 50, 2, 0.75, False, "average", "average", "cuda").to("cuda")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def train_model(self):

        for epoch_idx in tqdm(range(self.config.epochs)): # iterate through epoch
            acc_loss_train = 0
            
            for i in range(self.generator.number_of_batch): # iterate through scenegraphs
                
                data, label = next(self.generator)

                self.model.train()
                self.optimizer.zero_grad()
                               
                output = self.model.forward(data)

                loss_train = nn.CrossEntropyLoss()(output, torch.LongTensor(label).to("cuda"))

                loss_train.backward()

                self.optimizer.step()

                acc_loss_train += loss_train.detach().cpu().numpy()

            print('')
            print('Epoch: {:04d},'.format(epoch_idx), 'loss_train: {:.4f}'.format(acc_loss_train))
            print('')

    def predict_graph_classification(self):
        # take training set as testing data temporarily

        for i in range(self.generator.number_of_batch): # iterate through scenegraphs
            
            data, label = next(self.generator)
            
            self.model.eval()

            output = self.model.forward(data)

            acc_train = accuracy(output, torch.LongTensor(label))

            print('SceneGraph: {:04d}'.format(i), 'acc_train: {:.4f}'.format(acc_train.item()))


if __name__ == "__main__":
    gcn_trainer = GINTrainer(sys.argv[1:])
    gcn_trainer.build_model()
    gcn_trainer.train_model()
    gcn_trainer.predict_graph_classification()