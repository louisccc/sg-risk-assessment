import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.nn import Linear, LSTM
from torch_geometric.nn import RGCNConv, SAGPooling, TopKPooling, FastRGCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool


class MRGCN(nn.Module):
    
    def __init__(self, config):
        super(MRGCN, self).__init__()

        self.num_features = config.num_features
        self.num_relations = config.num_relations
        self.num_classes  = config.nclass
        self.num_layers = config.num_layers #defines number of RGCN conv layers.
        self.hidden_dim = config.hidden_dim
        self.hidden_dim2 = 20
        self.hidden_dim1 = 50

        self.pooling_type = config.pooling_type
        self.readout_type = config.readout_type
        self.temporal_type = config.temporal_type

        self.dropout = config.dropout
        self.conv = []
        self.conv.append(FastRGCNConv(self.num_features, self.hidden_dim, self.num_relations).to(config.device))

        dim = self.hidden_dim
        tot_dim = dim
        for i in range(1, self.num_layers):
            self.conv.append(FastRGCNConv(dim, dim // 2, self.num_relations).to(config.device))
            dim = dim // 2
            tot_dim += dim

        if self.pooling_type == "sagpool":
            self.pool1 = SAGPooling(tot_dim, ratio=config.pooling_ratio)
        elif self.pooling_type == "topk":
            self.pool1 = TopKPooling(tot_dim, ratio=config.pooling_ratio)

        if "lstm" in self.temporal_type:
            self.lstm = LSTM(self.hidden_dim1, self.hidden_dim2, batch_first=True)
            self.attn = Attention(self.hidden_dim2)

        self.fc1 = Linear(tot_dim, self.hidden_dim1)
        self.fc2 = Linear(self.hidden_dim2, self.num_classes)


    def forward(self, x, edge_index, edge_attr, batch=None):
        attn_weights = dict()
        outputs = []
        for i in range(self.num_layers):
            x = F.relu(self.conv[i](x, edge_index, edge_attr))
            outputs.append(x)
        x = torch.cat(outputs, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.pooling_type == "sagpool":
            x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        elif self.pooling_type == "topk":
            x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
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

        x = F.relu(self.fc1(x))
        
        if self.temporal_type == "mean":
            x = F.leaky_relu(x.mean(axis=0))
        elif self.temporal_type == "lstm_last":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = h.flatten()
        elif self.temporal_type == "lstm_sum":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = x_predicted.sum(dim=1).flatten()
        elif self.temporal_type == "lstm_attn":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x, attn_weights['lstm_attn_weights'] = self.attn(h.view(1,1,-1), x_predicted)
            x = x.flatten()
        else:
            pass
                
        return F.log_softmax(self.fc2(x), dim=-1)