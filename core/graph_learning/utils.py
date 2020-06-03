import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx
import pdb
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from matplotlib import pyplot as plt

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
    # get_roc_curve(output, labels, task)
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
def get_roc_curve(outputs, labels, task):
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
        plt.figure(figsize=(8,8))
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.title("Receiver Operating Characteristic for " + task)
        plt.plot([0,1],[0,1], linestyle='dashed')
        plt.plot(fpr,tpr, linewidth=2)
        plt.savefig("ROC_curve_"+task+".svg")
        # plt.show()
