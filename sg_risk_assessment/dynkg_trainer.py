import os, sys
import numpy as np
import pandas as pd
import random
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

import torch
import torch.optim as optim
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# sys.path.append(os.path.dirname(sys.path[0]))

from relation_extractor import Relations
from mrgcn import *
from metrics import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DynKGTrainer:

    def __init__(self, config):
        self.config = config
        self.args = config.args
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.summary_writer = SummaryWriter()

        self.best_val_loss = 99999
        self.best_epoch = 0
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.best_val_confusion = []
        self.best_val_f1 = 0
        self.best_val_mcc = -1.0
        self.best_val_acc_balanced = 0
        self.unique_clips = {}
        self.log = False

        if not self.config.cache_path.exists():
            raise Exception("The cache file does not exist.")    
        
    def init_dataset(self):
        self.training_data, self.testing_data, self.feature_list = self.build_scenegraph_dataset(self.config.cache_path, self.config.split_ratio, downsample=self.config.downsample, seed=self.config.seed, transfer_path=self.config.transfer_path)
        self.training_labels = [data['label'] for data in self.training_data]
        self.testing_labels = [data['label'] for data in self.testing_data]
        self.class_weights = torch.from_numpy(compute_class_weight('balanced', np.unique(self.training_labels), self.training_labels))
        print("Number of Sequences Included: ", len(self.training_data))
        print("Num Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
    
    def build_scenegraph_dataset(self, cache_path, train_to_test_ratio=0.3, downsample=False, seed=0, transfer_path=None):
        '''
            scenegraphs_sequence (gnn dataset):
                List of scenegraph data structures for evey clip
                Keys: {'sequence', 'label', 'folder_name', 'category'}
            feature_list:
                FILL IN DESCRIPTION HERE!
        '''
        dataset_file = open(cache_path, "rb")
        scenegraphs_sequence, feature_list = pkl.load(dataset_file)

        # Store driving categories and their frequencies
        self.unique_clips['all'] = 0
        for scenegraph in scenegraphs_sequence:
            self.unique_clips['all'] += 1
            if 'category' in scenegraph:
                category = scenegraph['category']
                if category in self.unique_clips:
                    self.unique_clips[category] += 1
                else:
                    self.unique_clips[category] = 1
            else:
                scenegraph['category'] = 'all'
                print('no category')
        print('Total dataset breakdown: {}'.format(self.unique_clips))        

        if transfer_path == None:
            class_0 = []
            class_1 = []

            for g in scenegraphs_sequence:
                if g['label'] == 0:
                    class_0.append(g)
                elif g['label'] == 1:
                    class_1.append(g)
                
            y_0 = [0]*len(class_0)
            y_1 = [1]*len(class_1)

            min_number = min(len(class_0), len(class_1))
            if downsample:
                modified_class_0, modified_y_0 = resample(class_0, y_0, n_samples=min_number)
            else:
                modified_class_0, modified_y_0 = class_0, y_0
                
            train, test, train_y, test_y = train_test_split(modified_class_0+class_1, modified_y_0+y_1, test_size=train_to_test_ratio, shuffle=True, stratify=modified_y_0+y_1, random_state=seed)
            return train, test, feature_list
        else: 
            test, _ = pkl.load(open(transfer_path, "rb"))
            return scenegraphs_sequence, test, feature_list 

    def build_model(self):
        self.config.num_features = len(self.feature_list)
        self.config.num_relations = max([r.value for r in Relations])+1
        if self.config.model == "mrgcn":
            self.model = MRGCN(self.config).to(self.config.device)
        elif self.config.model == "mrgin":
            self.model = MRGIN(self.config).to(self.config.device)
        else:
            raise Exception("model selection is invalid: " + self.config.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        if self.class_weights.shape[0] < 2:
            self.loss_func = nn.CrossEntropyLoss()
        else:    
           self.loss_func = nn.CrossEntropyLoss(weight=self.class_weights.float().to(self.config.device))
           
        self.config.wandb.watch(self.model, log="all")

    def train(self):
        tqdm_bar = tqdm(range(self.config.epochs))
        for epoch_idx in tqdm_bar: # iterate through epoch   
            acc_loss_train = 0
            self.sequence_loader = DataListLoader(self.training_data, batch_size=self.config.batch_size, shuffle=True)
            # TODO: Condense into one for loop
            for data_list in self.sequence_loader: # iterate through scenegraphs
                labels = torch.empty(0).long().to(self.config.device)
                outputs = torch.empty(0,2).to(self.config.device)
                self.model.train()
                self.optimizer.zero_grad()

                for sequence in data_list: # iterate through sequences
                    data, label = sequence['sequence'], sequence['label']
                    graph_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]
                    # data is a sequence that consists of serveral graphs 
                    self.train_loader = DataLoader(graph_list, batch_size=len(graph_list))
                    sequence = next(iter(self.train_loader)).to(self.config.device)
                    output, _ = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)
                    outputs = torch.cat([outputs, output.view(-1, 2)], dim=0)
                    labels  = torch.cat([labels, torch.LongTensor([label]).to(self.config.device)], dim=0)
                # import pdb; pdb.set_trace()
                loss_train = self.loss_func(outputs, labels)
                loss_train.backward()
                acc_loss_train += loss_train.detach().cpu().item() * len(data_list)
                self.optimizer.step()

            acc_loss_train /= len(self.training_data)
            tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
            
            if epoch_idx % self.config.test_step == 0:
                _, _, metrics, _ = self.evaluate(epoch_idx)
                self.summary_writer.add_scalar('Acc_Loss/train', metrics['train']['loss'], epoch_idx)
                self.summary_writer.add_scalar('Acc_Loss/train_acc', metrics['train']['acc'], epoch_idx)
                self.summary_writer.add_scalar('F1/train', metrics['train']['f1'], epoch_idx)
                # self.summary_writer.add_scalar('Confusion/train', metrics['train']['confusion'], epoch_idx)
                self.summary_writer.add_scalar('Precision/train', metrics['train']['precision'], epoch_idx)
                self.summary_writer.add_scalar('Recall/train', metrics['train']['recall'], epoch_idx)
                self.summary_writer.add_scalar('Auc/train', metrics['train']['auc'], epoch_idx)

                self.summary_writer.add_scalar('Acc_Loss/test', metrics['test']['loss'], epoch_idx)
                self.summary_writer.add_scalar('Acc_Loss/test_acc', metrics['test']['acc'], epoch_idx)
                self.summary_writer.add_scalar('F1/test', metrics['test']['f1'], epoch_idx)
                # self.summary_writer.add_scalar('Confusion/test', metrics['test']['confusion'], epoch_idx)
                self.summary_writer.add_scalar('Precision/test', metrics['test']['precision'], epoch_idx)
                self.summary_writer.add_scalar('Recall/test', metrics['test']['recall'], epoch_idx)
                self.summary_writer.add_scalar('Auc/test', metrics['test']['auc'], epoch_idx)

    def inference(self, X, y):
        labels = torch.LongTensor().to(self.config.device)
        outputs = torch.FloatTensor().to(self.config.device)
        # Dictionary storing (output, label) pair for all driving categories
        categories = dict.fromkeys(self.unique_clips)
        for key, val in categories.items():
            categories[key] = {'outputs': outputs, 'labels': labels}
        acc_loss_test = 0
        folder_names = []
        attns_weights = []
        node_attns = []
        inference_time = 0

        with torch.no_grad():
            for i in range(len(X)): # iterate through scenegraphs
                data, label, category = X[i]['sequence'], y[i], X[i]['category']
                data_list = [Data(x=g['node_features'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in data]
                self.test_loader = DataLoader(data_list, batch_size=len(data_list))
                sequence = next(iter(self.test_loader)).to(self.config.device)
                self.model.eval()

                #start = torch.cuda.Event(enable_timing=True)
                #end =  torch.cuda.Event(enable_timing=True)
                #start.record()
                output, attns = self.model.forward(sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch)
                #end.record()
                #torch.cuda.synchronize()
                inference_time += 0#start.elapsed_time(end)
                loss_test = self.loss_func(output.view(-1, 2), torch.LongTensor([label]).to(self.config.device))
                acc_loss_test += loss_test.detach().cpu().item()
                label = torch.tensor(label, dtype=torch.long).to(self.config.device)
                # store output, label statistics
                self.update_categorical_outputs(categories, output, label, category)
                
                folder_names.append(X[i]['folder_name'])
                if 'lstm_attn_weights' in attns:
                    attns_weights.append(attns['lstm_attn_weights'].squeeze().detach().cpu().numpy().tolist())
                if 'pool_score' in attns:
                    node_attn = {}
                    node_attn["original_batch"] = sequence.batch.detach().cpu().numpy().tolist()
                    node_attn["pool_perm"] = attns['pool_perm'].detach().cpu().numpy().tolist()
                    node_attn["pool_batch"] = attns['batch'].detach().cpu().numpy().tolist()
                    node_attn["pool_score"] = attns['pool_score'].detach().cpu().numpy().tolist()
                    node_attns.append(node_attn)
        
        sum_seq_len = 0
        num_risky_sequences = 0
        sequences = len(categories['all']['labels'])
        for indices in range(sequences):
            seq_output = categories['all']['outputs'][indices]
            label = categories['all']['labels'][indices]
            pred = torch.argmax(seq_output)        
            # risky clip
            if label == 1:
                num_risky_sequences += 1
                sum_seq_len += seq_output.shape[0]

        avg_risky_seq_len = sum_seq_len / num_risky_sequences 

        return  categories, \
                folder_names, \
                acc_loss_test/len(X), \
                avg_risky_seq_len, \
                inference_time, \
                attns_weights, \
                node_attns
    
    def evaluate(self, current_epoch=None):
        metrics = {}
        categories_train, \
        folder_names_train, \
        acc_loss_train, \
        train_avg_seq_len, \
        train_inference_time, \
        attns_train, \
        node_attns_train = self.inference(self.training_data, self.training_labels)

         # Collect metrics from all driving categories
        for category in self.unique_clips.keys():
            if category == 'all':
                metrics['train'] = get_metrics(categories_train['all']['outputs'], categories_train['all']['labels'])
                metrics['train']['loss'] = acc_loss_train
                metrics['train']['avg_seq_len'] = train_avg_seq_len
            else:
                metrics['train'][category] = get_metrics(categories_train[category]['outputs'], categories_train[category]['labels'])

        categories_test, \
        folder_names_test, \
        acc_loss_test, \
        val_avg_seq_len, \
        test_inference_time, \
        attns_test, \
        node_attns_test = self.inference(self.testing_data, self.testing_labels)
        
        # Collect metrics from all driving categories
        for category in self.unique_clips.keys():
            if category == 'all':
                metrics['test'] = get_metrics(categories_test['all']['outputs'], categories_test['all']['labels'])
                metrics['test']['loss'] = acc_loss_test
                metrics['test']['avg_seq_len'] = val_avg_seq_len
                metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / ((len(self.training_labels) + len(self.testing_labels)))
            else:
                metrics['test'][category] = get_metrics(categories_test[category]['outputs'], categories_test[category]['labels'])

        print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], "mcc:", metrics['train']['mcc'], \
              "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], "mcc:", metrics['test']['mcc'])
        
        #automatically save the model and metrics with the lowest validation loss
        self.update_best_metrics(metrics, current_epoch)
        metrics['best_epoch'] = self.best_epoch
        metrics['best_val_loss'] = self.best_val_loss
        metrics['best_val_acc'] = self.best_val_acc
        metrics['best_val_auc'] = self.best_val_auc
        metrics['best_val_conf'] = self.best_val_confusion
        metrics['best_val_f1'] = self.best_val_f1
        metrics['best_val_mcc'] = self.best_val_mcc
        metrics['best_val_acc_balanced'] = self.best_val_acc_balanced
            
        self.log2wandb(metrics)
        # NOTE: update code to support
        # self.save2csv(metrics) 
    
        return categories_train, categories_test, metrics, folder_names_train

    # Utilities
    def update_categorical_outputs(self, categories, outputs, labels, category):
        '''
            Aggregates output, label pairs for every driving category
            Based on inference setup, only one scenegraph_sequence is updated per call
        '''
        if category in categories:
            categories[category]['outputs'] = torch.cat([categories[category]['outputs'], torch.unsqueeze(outputs, dim=0)], dim=0)
            categories[category]['labels'] = torch.cat([categories[category]['labels'], torch.unsqueeze(labels, dim=0)], dim=0)
        # multi category
        if category != 'all': 
            category = 'all'
            categories[category]['outputs'] = torch.cat([categories[category]['outputs'], torch.unsqueeze(outputs, dim=0)], dim=0)
            categories[category]['labels'] = torch.cat([categories[category]['labels'], torch.unsqueeze(labels, dim=0)], dim=0)
        
        # reshape outputs
        for k, v in categories.items():
            categories[k]['outputs'] = categories[k]['outputs'].reshape(-1, 2)

    def update_best_metrics(self, metrics, current_epoch):
        if metrics['test']['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['test']['loss']
            self.best_epoch = current_epoch if current_epoch != None else self.config.epochs
            self.best_val_acc = metrics['test']['acc']
            self.best_val_auc = metrics['test']['auc']
            self.best_val_confusion = metrics['test']['confusion']
            self.best_val_f1 = metrics['test']['f1']
            self.best_val_mcc = metrics['test']['mcc']
            self.best_val_acc_balanced = metrics['test']['balanced_acc']
            #self.save_model()

    def save2csv(self, best_metrics):
        if not self.config.stats_path.exists():
            current_stats = pd.DataFrame(best_metrics, index=[0])
            current_stats.to_csv(str(self.config.stats_path), mode='w+', header=True, index=False, columns=list(best_metrics.keys()))
        else:
            best_stats = pd.read_csv(str(self.config.stats_path), header=0)
            best_stats = best_stats.reset_index(drop=True)
            replace_row = best_stats.loc[best_stats.args == str(self.args)]
            if(replace_row.empty):
                current_stats = pd.DataFrame(best_metrics, index=[0])
                current_stats.to_csv(str(self.config.stats_path), mode='a', header=False, index=False, columns=list(best_metrics.keys()))
            else:
                best_stats.iloc[replace_row.index] = pd.DataFrame(best_metrics, index=replace_row.index)
                best_stats.to_csv(str(self.config.stats_path), mode='w', header=True,index=False, columns=list(best_metrics.keys()))

    def save_model(self):
        """Function to save the model."""
        saved_path = Path(self.config.model_save_path).resolve()
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        torch.save(self.model.state_dict(), str(saved_path))
        with open(os.path.dirname(saved_path) + "/model_parameters.txt", "w+") as f:
            f.write(str(self.config))
            f.write('\n')
            f.write(str(' '.join(sys.argv)))

    def load_model(self):
        """Function to load the model."""
        saved_path = Path(self.config.model_load_path).resolve()
        if saved_path.exists():
            self.build_model()
            self.model.load_state_dict(torch.load(str(saved_path)))
            self.model.eval()
   
    def log2wandb(self, metrics):
        '''
            Log metrics from all driving categories
        '''
        for category in self.unique_clips.keys():
            if category == 'all':
                log_wandb(self.config.wandb, metrics)
            else:
                log_wandb_categories(self.config.wandb, metrics, id=category)

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

#~~~~~~~~~~Scoring Metrics~~~~~~~~~~
#note: these scoring metrics only work properly for binary classification use cases (graph classification, dyngraph classification) 
def get_auc(outputs, labels):
    try:    
        labels = encode_onehot(labels.numpy().tolist(), 2) #binary labels
        auc = roc_auc_score(labels, outputs.numpy(), average="micro")
    except ValueError as err: 
        print("error calculating AUC: ", err)
        auc = 0.0
    return auc

#NOTE: ROC curve is only generated for positive class (risky label) confidence values 
#render parameter determines if the figure is actually generated. If false, it saves the values to a csv file.
def get_roc_curve(outputs, labels, render=False):
    risk_scores = []
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