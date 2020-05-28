import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


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




