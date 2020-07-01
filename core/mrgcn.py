import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.nn import Linear, LSTM
from torch_geometric.nn import RGCNConv, SAGPooling, TopKPooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool


class MRGCN(nn.Module):
    
    def __init__(self, config):
        super(MRGCN, self).__init__()

        self.num_features = config.num_features
        self.num_relations = config.num_relations
        self.num_classes  = config.nclass
        #self.num_layers = config.num_layers #TODO: remove if unnecessary
        self.hidden_dim = config.hidden_dim

        self.gin_convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.pooling_type = config.pooling_type
        self.readout_type = config.readout_type
        self.temporal_type = config.temporal_type

        self.dropout = config.dropout

        self.conv1 = RGCNConv(self.num_features, self.hidden_dim, self.num_relations, num_bases=30)
        self.conv2 = RGCNConv(self.hidden_dim,   self.hidden_dim, self.num_relations, num_bases=30)

        if self.pooling_type == "sagpool":
            self.pool1 = SAGPooling(self.hidden_dim, ratio=0.8)
        elif self.pooling_type == "topk":
            self.pool1 = TopKPooling(self.hidden_dim, ratio=0.8)
        # elif self.pooling_type == "asa":
        #     self.pool1 = ASAPooling(self.hidden_dim, ratio=0.8)

        if "lstm" in self.temporal_type:
            self.lstm = LSTM(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True)
            self.attn = Attention(self.hidden_dim * 2)
            self.fc3 = Linear(self.hidden_dim * 2, self.hidden_dim)

        self.fc1 = Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.num_classes)


    def forward(self, x, edge_index, edge_attr, batch=None):
        attn_weights = dict()

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        if self.pooling_type == "sagpool":
            x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        elif self.pooling_type == "topk":
            x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        # elif self.pooling_type == "asa":
        #     x, edge_index, _, batch, attn_weights['pool_perm'] = self.pool1(x, edge_index, batch=batch)
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
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)