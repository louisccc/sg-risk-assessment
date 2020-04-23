from __future__ import division
from __future__ import print_function

import time, pdb
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from pygcn.utils import load_data, accuracy, load_scene_data
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
    
adjs, features, labels = load_scene_data()

n_features = next(iter(features.values())).shape[1]

#not used:
idx_train = range(0)
idx_val = range(0)
idx_test = range(0)


#returns an embedding for each node (unsupervised)
model = GCN(nfeat=n_features,
            nhid=args.hidden,
            nclass=n_features,
            dropout=args.dropout)

# Model and optimizer
#model = GCN(nfeat=features.shape[1],
#            nhid=args.hidden,
#            nclass=labels.max().item() + 1,
#            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adjs = adjs.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(features, adj, labels):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    #acc_train = accuracy(output[idx_train], labels[idx_train])
    #loss_train.backward()
    optimizer.step()
    
#    if not args.fastmode:
#        # Evaluate validation set performance separately,
#        # deactivates dropout during validation run.
#        model.eval()
#        output = model(features, adj)
#
#    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#    acc_val = accuracy(output[idx_val], labels[idx_val])
#    print('Epoch: {:04d}'.format(1),
#          'loss_train: {:.4f}'.format(loss_train.item()),
#          'acc_train: {:.4f}'.format(acc_train.item()))
#          'loss_val: {:.4f}'.format(loss_val.item()),
#          'acc_val: {:.4f}'.format(acc_val.item()),
#          'time: {:.4f}s'.format(time.time() - t))
    return output

#def test():
#    model.eval()
#    output = model(features, adj)
#    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#    acc_test = accuracy(output[idx_test], labels[idx_test])
#    print("Test set results:",
#          "loss= {:.4f}".format(loss_test.item()),
#          "accuracy= {:.4f}".format(acc_test.item()))


#generate embeddings for every scene graph and concatenate into a huge dataframe for projector evaluation
result_embeddings = pd.DataFrame()
features = list(features.values())
adjs =  list(adjs.values())
labels =  list(labels.values())
for i in range(len(features)):
    output = train(features[i], adjs[i], labels[i])
    result_embeddings = pd.concat([result_embeddings, pd.DataFrame(output.detach().numpy().reshape(output.size()[0],n_features))], axis=0, ignore_index=True)

pdb.set_trace()
labels = np.concatenate(labels)
result_embeddings.to_csv("result_embeddings.tsv", sep="\t", header=False, index=False)
pd.DataFrame(labels).to_csv('meta.tsv', sep='\t', header=False, index=False)


# Train model
#t_total = time.time()
#for epoch in range(args.epochs):
#    train(epoch)
#print("Optimization Finished!")
#print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
#test()
