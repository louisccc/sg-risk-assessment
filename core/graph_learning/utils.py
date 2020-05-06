import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx
import pdb
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_curve

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
    
    outputs = torch.cat(outputs)
    outputs = pd.DataFrame(outputs.cpu().detach().numpy())
    labels = np.concatenate(labels) if len(labels) > 1 else labels
    pd.DataFrame(labels).to_csv(output_dir / str(filename + "_labels.tsv"), sep='\t', header=False, index=False)
    outputs.to_csv(output_dir / str(filename + "_outputs.tsv"), sep="\t", header=False, index=False)


#~~~~~~~~~~Scoring Metrics~~~~~~~~~~

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
    
def get_scoring_metrics(output, labels):
    pdb.set_trace()
    preds, labels = get_predictions(output, labels)
    metrics = defaultdict()
    metrics['acc'] = overall_accuracy(preds, labels)
    metrics['f1'] = get_f1_score(preds, labels)
    metrics['confusion'] = str(confusion_matrix(preds, labels))
    
    return metrics
    
def get_predictions(outputs, labels):
    if(len(outputs) > 1 and len(labels) > 1):
        labels = torch.LongTensor(np.concatenate(labels))
        preds = torch.cat(outputs).max(1)[1].type_as(labels)
    else:
        labels = torch.LongTensor(labels)
        preds = torch.cat(outputs).max(0)[1].type_as(labels)
    return preds, labels
    
def overall_accuracy(preds, labels):    
    if(len(preds) > 1 and len(labels) > 1):
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct.item() / len(labels)
    else:
        #if only one value in output (edge case)
        correct = preds.eq(labels).double()
        return correct.item()
        
#TODO: adapt for multi-class scoring as well
def get_f1_score(preds, labels):
    return f1_score(labels, preds)
    
def get_confusion_matrix(preds, labels):
    return confusion_matrix(labels, preds)
    
def get_precision(preds, labels):
    pass
    
def get_recall(labels, labels):
    pass
    
    



