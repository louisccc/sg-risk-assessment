import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torchnlp.nn import Attention
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling, ASAPooling
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, pooling_type=None, readout_type=None, temporal_type=None):
        super(GCN, self).__init__()

        self.pooling_type = pooling_type
        # switch between average/max/mean/sort.
        self.readout_type = readout_type

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout

        self.pooling_type = pooling_type
        self.readout_type = readout_type
        self.temporal_type = temporal_type

        if self.pooling_type == "sagpool":
            self.pool1 = SAGPooling(nclass, ratio=0.8)
        elif self.pooling_type == "topk":
            self.pool1 = TopKPooling(nclass, ratio=0.8)
        elif self.pooling_type == "asa":
            self.pool1 = ASAPooling(nclass, ratio=0.8)
        
        if "lstm" in self.temporal_type:
            self.lstm = nn.LSTM(nclass, nhid, batch_first=True, bidirectional=True)
            self.attn = Attention(self.hidden_dim * 2)
            self.fc1 = nn.Linear(2*nhid, nclass)

    def forward(self, x, edge_index, batch=None):
        ''' graphs_in_batch is a list of graph instances; '''
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)

        if self.pooling_type == "sagpool":
            x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, batch=batch)
        elif self.pooling_type == "topk":
            x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, batch=batch)
        elif self.pooling_type == "asa":
            x, edge_index, _, batch, perm = self.pool1(x, edge_index, batch=batch)

        if self.readout_type == "add":
            x = global_add_pool(x, batch)
        elif self.readout_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout_type == "max":
            x = global_max_pool(x, batch)
        elif self.readout_type == "sort":
            x = global_sort_pool(x, batch, k=100)
        else:
            pass
        
        if self.temporal_type == "mean":
            x = x.mean(axis=0)
        elif self.temporal_type == "lstm_last":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = F.relu(self.fc1(h.flatten()))
        elif self.temporal_type == "lstm_sum":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = F.relu(self.fc1(x_predicted.sum(dim=1).flatten()))
        elif self.temporal_type == "lstm_attn":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x, weights = self.attn(h.view(1,1,-1), x_predicted)
            x = self.fc1(x.flatten())
        else:
            pass
        
        if self.temporal_type:
            return F.log_softmax(x, dim=0)

        return F.log_softmax(x, dim=1)
