import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx
import pdb
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score

#returns onehot version of labels (unused)
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


#copied from https://github.com/tkipf/pygcn/tree/master/data/cora to load cora dataset
def load_cora_data(path="../../input/cora/", dataset="cora"):
    """Load citation network dataset"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
    
    
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

#generate TSV output file from outputs and labels
def save_outputs(output_dir, outputs, labels, filename):
    outputs = pd.DataFrame(outputs.cpu().detach().numpy())
    labels = labels.flatten()
    pd.DataFrame(labels).to_csv(output_dir / str(filename + "_labels.tsv"), sep='\t', header=False, index=False)
    outputs.to_csv(output_dir / str(filename + "_outputs.tsv"), sep="\t", header=False, index=False)


#~~~~~~~~~~Scoring Metrics~~~~~~~~~~
#note: these scoring metrics only work properly for binary classification use cases (graph classification, dyngraph classification) 

#used for output of validation accuracy after each epoch
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    

def get_scoring_metrics(output, labels, task):
    pdb.set_trace()
    preds, labels = get_predictions(output, labels)
    metrics = defaultdict()
    metrics['acc'] = accuracy_score(labels, preds)
    metrics['f1'] = f1_score(labels, preds)
    metrics['confusion'] = str(confusion_matrix(labels, preds))
    metrics['precision'] = precision_score(labels, preds)
    metrics['recall'] = recall_score(labels, preds)
    metrics['auc'] = get_auc(output, labels)
    return metrics
    
def get_predictions(outputs, labels):
    labels = torch.LongTensor(labels)
    preds = outputs.max(1)[1].type_as(labels)
    return preds, labels

def get_auc(outputs, labels):
    try:
        auc = roc_auc_score(labels.numpy(), outputs.numpy())
    except ValueError: #thrown when labels only has one class
        print("Labels only has a single class in its values. AUC is 0.")
        auc = 0.0
    return auc