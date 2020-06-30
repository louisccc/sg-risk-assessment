import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import pandas as pd
import random
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from matplotlib import pyplot as plt

from core.scene_graph import SceneGraphSequenceGenerator, build_scenegraph_dataset
from core.relation_extractor import Relations
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from core.mrgcn import *
from torch_geometric.data import Data, DataLoader
from sklearn.utils.class_weight import compute_class_weight

class Config:
    '''Argument Parser for script to train scenegraphs.'''
    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for training the scene graph using GCN.')
        self.parser.add_argument('--input_path', type=str, default="../input/synthesis_data/lane-change/", help="Path to code directory.")
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='The initial learning rate for GCN.')
        self.parser.add_argument('--seed', type=int, default=random.randint(0,2**32), help='Random seed.')
        self.parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--hidden', type=int, default=200, help='Number of hidden units.')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--nclass', type=int, default=2, help="The number of classes for dynamic graph classification.")
        self.parser.add_argument('--batch_size', type=int, default=32, help='Number of graphs in a batch.')
        self.parser.add_argument('--device', type=str, default="cpu", help='The device to run on models (cuda or cpu) cpu in default.')
        self.parser.add_argument('--test_step', type=int, default=10, help='Number of epochs before testing the model.')
        self.parser.add_argument('--model', type=str, default="mrgcn", help="Model to be used intrinsically.")
        self.parser.add_argument('--num_layers', type=int, default=5, help="Number of layers in the neural network.")
        self.parser.add_argument('--hidden_dim', type=int, default=32, help="Hidden dimension in GIN.")
        self.parser.add_argument('--pooling_type', type=str, default="sagpool", help="Graph pooling type.")
        self.parser.add_argument('--readout_type', type=str, default="mean", help="Readout type.")
        self.parser.add_argument('--temporal_type', type=str, default="lstm_sum", help="Temporal type.")

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()


class DynKGTrainer:

    def __init__(self, args):
        self.config = Config(args)

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # load carla cheating scene graph txts into memory 
        self.training_data, self.testing_data, self.feature_list = build_scenegraph_dataset(self.config.input_base_dir)
        self.training_labels = [data['label'] for data in self.training_data]
        self.testing_labels = [data['label'] for data in self.testing_data]
        self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.training_labels), self.training_labels))
        print("Number of Sequences Included: ", len(self.training_data))
        print("Num Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))


    def build_model(self):
        self.config.num_features = len(self.feature_list)
        self.config.num_relations = max([r.value for r in Relations])+1
        self.model = MRGCN(self.config).to(self.config.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        if self.class_weights.shape[0] < 2:
            self.loss_func = nn.CrossEntropyLoss()
        else:    
           self.loss_func = nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.device))

    def train(self):

        for epoch_idx in tqdm(range(self.config.epochs)): # iterate through epoch
            acc_loss_train = 0
            for i in range(len(self.training_data)): # iterate through scenegraphs
                data, label = self.training_data[i]['sequence'], self.training_labels[i]

                data_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]
                
                self.train_loader = DataLoader(data_list, batch_size=len(data_list))
                sequence = next(iter(self.train_loader)).to(self.config.device)

                self.model.train()
                self.optimizer.zero_grad()

                output = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)

                loss_train = self.loss_func(output.view(-1, 2), torch.LongTensor([label]).to(self.config.device))
                loss_train.backward()

                self.optimizer.step()
                acc_loss_train += loss_train.detach().cpu().item()

            print('')
            print('Epoch: {:04d},'.format(epoch_idx), 'loss_train: {:.4f}'.format(acc_loss_train))
            print('')

            if epoch_idx % self.config.test_step == 0:
                self.evaluate()

    def inference(self, testing_data, testing_labels):
        acc_predict = []
        labels = []
        outputs = []
        for i in range(len(testing_data)): # iterate through scenegraphs
            data, label = testing_data[i]['sequence'], testing_labels[i]
            
            data_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]

            self.test_loader = DataLoader(data_list, batch_size=len(data_list))
            sequence = next(iter(self.train_loader)).to(self.config.device)

            self.model.eval()
            output = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)
            
            # print(output, label)
            acc_test = accuracy(output.view(-1, 2), torch.LongTensor([label]).to(self.config.device))
            acc_predict.append(acc_test.detach().cpu().item())

            outputs.append(output.detach().cpu())
            labels.append(label)

            # print('Dynamic SceneGraph: {:04d}'.format(i), 'acc_test: {:.4f}'.format(acc_test.item()))

        return outputs, labels, acc_predict

    def evaluate(self):
        
        outputs, labels, acc_predict = self.inference(self.training_data, self.training_labels)
        print('Dynamic SceneGraph training precision', sum(acc_predict) / len(acc_predict))

        outputs, labels, acc_predict = self.inference(self.testing_data, self.testing_labels)
        print('Dynamic SceneGraph testing precision', sum(acc_predict) / len(acc_predict))
        
        outputs = torch.cat(outputs).reshape(-1,2).detach()
       
        return outputs, np.array(labels).flatten()


#returns onehot version of labels. can specify n_classes to force onehot size.
def encode_onehot(labels, n_classes=None):
    if(n_classes):
        classes = set(range(n_classes))
    else:
        classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
    
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#~~~~~~~~~~Scoring Metrics~~~~~~~~~~
#note: these scoring metrics only work properly for binary classification use cases (graph classification, dyngraph classification) 

#used for output of validation accuracy after each epoch
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    

def get_scoring_metrics(output, labels, task):
    preds, labels = get_predictions(output, labels)
    metrics = defaultdict()
    output = torch.FloatTensor(output)
    #import pdb;pdb.set_trace()
    metrics['acc'] = accuracy_score(labels, preds)
    metrics['f1'] = f1_score(labels, preds, average="micro")
    metrics['confusion'] = str(confusion_matrix(labels, preds))
    metrics['precision'] = precision_score(labels, preds, average="micro")
    metrics['recall'] = recall_score(labels, preds, average="micro")
    metrics['auc'] = get_auc(output, labels, task)
    metrics['label_distribution'] = str(np.unique(labels, return_counts=True)[1])
    get_roc_curve(output, labels, task, render=False)
    return metrics
    
def get_predictions(outputs, labels):
    labels = torch.LongTensor(labels)
    outputs = torch.FloatTensor(outputs)
    preds = outputs.max(1)[1].type_as(labels)
    return preds, labels


def get_auc(outputs, labels, task):
    try:
        if(task == "node_classification"):
            labels = encode_onehot(labels.numpy().tolist(), 8) #multiclass labels
            auc = roc_auc_score(labels, outputs.numpy(), average="weighted")
        else:
            labels = encode_onehot(labels.numpy().tolist(), 2) #binary labels
            auc = roc_auc_score(labels, outputs.numpy(), average="micro")
    except ValueError as err: 
        print("error calculating AUC: ", err)
        auc = 0.0
    return auc

#NOTE: ROC curve is only generated for positive class (risky label) confidence values 
#render parameter determines if the figure is actually generated. If false, it saves the values to a csv file.
def get_roc_curve(outputs, labels, task, render=False):
    if task == "node_classification":
        print("Node classification ROC not implemented")
        return None
    else:
        risk_scores = []
       #pdb.set_trace()
        outputs = preprocessing.normalize(outputs.numpy(), axis=0)
        for i in outputs:
            risk_scores.append(i[1])
        fpr, tpr, thresholds = roc_curve(labels.numpy(), risk_scores)
        roc = pd.DataFrame()
        roc['fpr'] = fpr
        roc['tpr'] = tpr
        roc['thresholds'] = thresholds
        roc.to_csv("ROC_data_"+task+".csv")

        if(render):
            plt.figure(figsize=(8,8))
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.ylabel("TPR")
            plt.xlabel("FPR")
            plt.title("Receiver Operating Characteristic for " + task)
            plt.plot([0,1],[0,1], linestyle='dashed')
            plt.plot(fpr,tpr, linewidth=2)
            plt.savefig("ROC_curve_"+task+".svg")
