import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.nn import Linear, LSTM
from torch_geometric.nn import RGCNConv, SAGPooling, TopKPooling, FastRGCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool

from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax


class RGCNSAGPooling(torch.nn.Module):
    def __init__(self, in_channels, num_relations, ratio=0.5, min_score=None,
                 multiplier=1, nonlinearity=torch.tanh, rgcn_func="FastRGCNConv", **kwargs):
        super(RGCNSAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = FastRGCNConv(in_channels, 1, num_relations, **kwargs) if rgcn_func=="FastRGCNConv" else RGCNConv(in_channels, 1, num_relations, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()


    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index, edge_attr).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]


    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn.__class__.__name__,
            self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)

class MRGCN(nn.Module):
    
    def __init__(self, config):
        super(MRGCN, self).__init__()

        self.num_features = config.num_features
        self.num_relations = config.num_relations
        self.num_classes  = config.nclass
        self.num_layers = config.num_layers #defines number of RGCN conv layers.
        self.hidden_dim = config.hidden_dim
        self.layer_spec = None if config.layer_spec == None else list(map(int, config.layer_spec.split(',')))
        self.lstm_dim1 = config.lstm_input_dim
        self.lstm_dim2 = config.lstm_output_dim
        self.rgcn_func = FastRGCNConv if config.conv_type == "FastRGCNConv" else RGCNConv
        self.pooling_type = config.pooling_type
        self.readout_type = config.readout_type
        self.temporal_type = config.temporal_type

        self.dropout = config.dropout
        self.conv = []
        total_dim = 0

        if self.layer_spec == None:
            self.conv.append(self.rgcn_func(self.num_features, self.hidden_dim, self.num_relations).to(config.device))
            total_dim += self.hidden_dim
            for i in range(1, self.num_layers):
                self.conv.append(self.rgcn_func(self.hidden_dim, self.hidden_dim, self.num_relations).to(config.device))
                total_dim += self.hidden_dim
        
        else:
            print("using layer specification and ignoring hidden_dim parameter.")
            print("layer_spec: " + str(self.layer_spec))
            self.conv.append(self.rgcn_func(self.num_features, self.layer_spec[0], self.num_relations).to(config.device))
            total_dim += self.layer_spec[0]
            for i in range(1, self.num_layers):
                self.conv.append(self.rgcn_func(self.layer_spec[i-1], self.layer_spec[i], self.num_relations).to(config.device))
                total_dim += self.layer_spec[i]

            self.hidden_dim = self.layer_spec[-1] #setting the hidden dims of all later layers with last layer size of layer spec.

        if self.pooling_type == "sagpool":
            self.pool1 = RGCNSAGPooling(total_dim, self.num_relations, ratio=config.pooling_ratio, rgcn_func=config.conv_type)
        elif self.pooling_type == "topk":
            self.pool1 = TopKPooling(total_dim, ratio=config.pooling_ratio)

        self.fc1 = Linear(total_dim, self.lstm_dim1)
        
        if "lstm" in self.temporal_type:
            self.lstm = LSTM(self.lstm_dim1, self.lstm_dim2, batch_first=True)
            self.attn = Attention(self.lstm_dim2)
        
        self.fc2 = Linear(self.lstm_dim2, self.num_classes)


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