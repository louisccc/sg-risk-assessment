import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, edge_index):
        ''' graphs_in_batch is a list of graph instances; '''
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_Graph(nn.Module):
    ''' graph level model '''
    def __init__(self, nfeat, nhid, nclass, dropout, pooling_type):
        super(GCN_Graph, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

        # switch between average/max/mean/sort.
        self.pooling_type = pooling_type

    def forward(self, x, edge_index, batch):
        ''' graphs_in_batch is a list of graph instances; '''
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        
        if self.pooling_type == "add":
            x = global_add_pool(x, batch)
        elif self.pooling_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            x = global_max_pool(x, batch)
        elif self.pooling_type == "sort":
            x = global_sort_pool(x, batch, k=100)
        else:
            pass

        return F.log_softmax(x, dim=1)

class GCN_Graph_Sequence(nn.Module):
    ''' graph_sequence level model '''
    def __init__(self, nfeat, nhid, nclass, dropout, pooling_type, temporal_type):
        super(GCN_Graph_Sequence, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

        # switch between average/max/mean/sort.
        self.pooling_type = pooling_type

        self.temporal_type = temporal_type

    def forward(self, x, edge_index, batch):
        ''' graphs_in_batch is a list of graph instances; indicating a graph_sequence. '''
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        
        if self.pooling_type == "add":
            x = global_add_pool(x, batch)
        elif self.pooling_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            x = global_max_pool(x, batch)
        elif self.pooling_type == "sort":
            x = global_sort_pool(x, batch, k=100)
        else:
            pass

        if self.temporal_type == "mean":
            x = x.mean(axis=0)
        else: # lstm or others.
            pass
        
        return F.log_softmax(x, dim=0)