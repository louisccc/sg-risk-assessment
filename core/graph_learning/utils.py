import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx
import pdb
from collections import defaultdict

#returns onehot version of labels (unused)
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


#gets a list of all feature labels for all scenegraphs
def get_feature_list(scenegraphs, num_classes):
    all_attrs = set()
    for scenegraph in scenegraphs:
        for entity in scenegraph.entity_nodes:
            all_attrs.update(entity.attr.keys())
            
    final_attr_list = all_attrs.copy()
    for attr in all_attrs:
        if attr in ["location", "rotation", "velocity", "ang_velocity"]:
            final_attr_list.discard(attr)
            final_attr_list.update([attr+"_x", attr+"_y", attr+"_z"]) #add 3 columns for vector values
    for i in range(num_classes):
        final_attr_list.add("type_"+str(i)) #create 1hot class labels
    final_attr_list.discard("name") #remove node name as it is not needed sice we have class labels
    return sorted(final_attr_list)



#generates a list of node embeddings based on the node attributes and the feature list
#TODO: convert all non-numeric features to numeric datatypes
def create_node_embeddings(scenegraph, feature_list):
    rows = []
    for node in scenegraph.entity_nodes:
        row = defaultdict()
        for attr in node.attr:
            if attr in ["location", "rotation", "velocity", "ang_velocity"]:
                row[attr+"_x"] = node.attr[attr][0]
                row[attr+"_y"] = node.attr[attr][1]
                row[attr+"_z"] = node.attr[attr][2]
            elif attr == "is_junction": #binarizing junction label
                row[attr] = 1 if node.attr==True else 0
            elif attr == "name": #dont add name to embedding
                continue
            else:
                row[attr] = node.attr[attr]
        row['type_'+str(node.type)] = 1 #assign 1hot class label
        rows.append(row)
    #pdb.set_trace()
    embedding = pd.DataFrame(data=rows, columns=feature_list)
    embedding = embedding.fillna(value=0) #fill in NaN with zeros
    
    return embedding


#get adjacency matrix for scenegraph in scipy.sparse CSR matrix format
def get_adj_matrix(scenegraph):
    adj = nx.convert_matrix.to_scipy_sparse_matrix(scenegraph.g, nodelist=scenegraph.entity_nodes)
    return adj
    
    
    
    

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
    
    
