import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU, LSTM
from torch_geometric.nn import GINConv, SAGPooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool

class GIN(nn.Module):
    
    def __init__(self, args, num_features, num_classes, num_layers, pooling_type, readout_type, temporal_type):
        super(GIN, self).__init__()

        self.num_features = num_features
        self.num_classes  = num_classes
        self.num_layers = num_layers
        self.hidden_dim = 32

        self.gin_convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.pooling_type = pooling_type
        self.readout_type = readout_type
        self.temporal_type = temporal_type

        if self.pooling_type == "sagpool":
            self.pool1 = SAGPooling(self.hidden_dim, ratio=0.8)

        if self.temporal_type == "lstm":
            self.lstm = LSTM(self.num_classes, self.hidden_dim, batch_first=True, bidirectional=True)
            self.reduce_h = Linear(self.hidden_dim * 2, self.num_classes)

        for layer in range(self.num_layers-1):
            if layer == 0:
                nn = Sequential(Linear(num_features, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            else:
                nn = Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            self.gin_convs.append(GINConv(nn))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_dim))

        self.fc1 = Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.num_classes)

    def forward(self, x, edge_index, batch=None):
        for layer in range(self.num_layers-1):
            x = F.relu(self.gin_convs[layer](x, edge_index))
            x = self.batch_norms[layer](x)

        if self.pooling_type == "sagpool":
            x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, batch=batch)

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
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        if self.temporal_type == "mean":
            x = x.mean(axis=0)
        elif self.temporal_type == "lstm":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = F.relu(self.reduce_h(h.flatten()))
        else:
            pass

        return F.log_softmax(x, dim=-1)


class GIN_Graph(nn.Module):
    
    def __init__(self, args, num_features, num_classes, num_layers, pooling_type, readout_type):
        super(GIN_Graph, self).__init__()

        self.num_features = num_features
        self.num_classes  = num_classes
        self.num_layers = num_layers
        self.hidden_dim = 32

        self.gin_convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.pooling_type = pooling_type
        self.readout_type = readout_type

        for layer in range(self.num_layers-1):
            if layer == 0:
                nn = Sequential(Linear(num_features, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            else:
                nn = Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            self.gin_convs.append(GINConv(nn))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_dim))

        if self.pooling_type == "sagpool":
            self.pool1 = SAGPooling(self.hidden_dim, ratio=0.8)

        self.fc1 = Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.num_classes)

    def forward(self, x, edge_index, batch):
        for layer in range(self.num_layers-1):
            x = F.relu(self.gin_convs[layer](x, edge_index))
            x = self.batch_norms[layer](x)

        if self.pooling_type == "sagpool":
            x, edge_index, _, batch, perm, score = self.pool1(x, edge_index)

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
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class GIN_Graph_Sequence(nn.Module):
    
    def __init__(self, args, num_features, num_classes, pooling_type, readout_type, temporal_type, num_layers):
        super(GIN_Graph_Sequence, self).__init__()

        self.num_features = num_features
        self.num_classes  = num_classes
        self.num_layers = num_layers
        self.hidden_dim = 32

        self.pooling_type = pooling_type
        self.readout_type = readout_type
        self.temporal_type = temporal_type
        
        self.gin_convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                nn = Sequential(Linear(num_features, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            else:
                nn = Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
            self.gin_convs.append(GINConv(nn))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_dim))

        self.fc1 = Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.num_classes)

        if self.pooling_type == "sagpool":
            self.pool1 = SAGPooling(self.hidden_dim, ratio=0.8)

        self.lstm = LSTM(self.num_classes, self.hidden_dim, batch_first=True, bidirectional=True)
        self.reduce_h = Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, x, edge_index, batch):
        for layer in range(self.num_layers-1):
            x = F.relu(self.gin_convs[layer](x, edge_index))
            x = self.batch_norms[layer](x)

        if self.pooling_type == "sagpool":
            x, edge_index, _, batch, perm, score = self.pool1(x, edge_index)

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
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        if self.temporal_type == "mean":
            x = x.mean(axis=0)
        elif self.temporal_type == "lstm":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = F.relu(self.reduce_h(h.flatten()))
        else:
            pass

        return F.log_softmax(x, dim=-1)