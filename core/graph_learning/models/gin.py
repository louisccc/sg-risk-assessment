import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.nn import Sequential, Linear, ReLU, LSTM
from torch_geometric.nn import GINConv, SAGPooling, TopKPooling, ASAPooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool


class GIN(nn.Module):
    
    def __init__(self, args, num_features, num_classes, num_layers, hidden_dim, pooling_type=None, readout_type=None, temporal_type=None):
        super(GIN, self).__init__()

        self.num_features = num_features
        self.num_classes  = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gin_convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.pooling_type = pooling_type
        self.readout_type = readout_type
        self.temporal_type = temporal_type

        for layer in range(self.num_layers-1):
            if layer == 0:
                nn = Sequential(Linear(num_features, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            else:
                nn = Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            self.gin_convs.append(GINConv(nn))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_dim))

        if self.pooling_type == "sagpool":
            self.pool1 = SAGPooling(self.hidden_dim, ratio=0.8)
        elif self.pooling_type == "topk":
            self.pool1 = TopKPooling(self.hidden_dim, ratio=0.8)
        elif self.pooling_type == "asa":
            self.pool1 = ASAPooling(self.hidden_dim, ratio=0.8)

        if "lstm" in self.temporal_type:
            self.lstm = LSTM(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True)
            self.attn = Attention(self.hidden_dim * 2)
            self.fc3 = Linear(self.hidden_dim * 2, self.hidden_dim)

        self.fc1 = Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.num_classes)


    def forward(self, x, edge_index, batch=None):
        attn_weights = dict()

        for layer in range(self.num_layers-1):
            x = F.relu(self.gin_convs[layer](x, edge_index))
            x = self.batch_norms[layer](x)

        if self.pooling_type == "sagpool":
            x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, batch=batch)
        elif self.pooling_type == "topk":
            x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, batch=batch)
        elif self.pooling_type == "asa":
            x, edge_index, _, batch, attn_weights['pool_perm'] = self.pool1(x, edge_index, batch=batch)
        else: 
            pass

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
            x = F.relu(self.fc3(h.flatten()))
        elif self.temporal_type == "lstm_sum":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = F.relu(self.fc3(x_predicted.sum(dim=1).flatten()))
        elif self.temporal_type == "lstm_attn":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x, attn_weights['lstm_attn_weights'] = self.attn(h.view(1,1,-1), x_predicted)
            x = self.fc3(x.flatten())
        else:
            pass
            
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)
