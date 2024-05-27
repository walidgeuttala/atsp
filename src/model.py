import torch
from torch import nn, optim
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    GENConv,
    DirGNNConv,
    FAConv,
    JumpingKnowledge,
    Sequential,
    MixHopConv
)


from dataset.data_utils import get_norm_adj


def get_conv(conv_type, input_dim, output_dim, alpha):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "gen":
        return GENConv(input_dim, output_dim, aggr='powermean', t=1.0, learn_t=True, num_layers=2, norm='layer') 
    elif conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-sage":
        return DirSageConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gat":
        return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    elif conv_type == 'dir-gen':
        return DirGNNConv(GENConv(input_dim, output_dim, aggr='powermean', t=1.0, learn_t=True, num_layers=2, norm='layer'))
    elif conv_type == 'dir-fa':
        return FAConv(-1, )
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x, edge_index=None):
        if edge_index is not None:
            y = self.module(x, edge_index).view(-1, x.size(-1))
        else:
            y = self.module(x)
        return x + y


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, first_layer=False):
        super(AttentionLayer, self).__init__()

        if first_layer:
            self.message_passing = MixHopConv(embed_dim, embed_dim, [0 ,1, 2], False, True)
                                
        else:
            self.message_passing = SkipConnection(
                                MixHopConv(embed_dim, embed_dim, [0 ,1, 2], False, True)
                                )
        if embed_dim // n_heads * n_heads != embed_dim:
            print('wrong')
        
        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(embed_dim*3),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim*3, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embed_dim)
                ),
            ),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x, edge_index):
        h = self.message_passing(x, edge_index)
        h = self.feed_forward(h)
        return h
    
class EdgePropertyPredictionModel(nn.Module):
    def __init__(
            self,
            num_features,
            hidden_dim,
            num_classes,
            num_layers,
            n_heads=1,
    ):
        super(EdgePropertyPredictionModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.embed_layer = Linear(num_features, hidden_dim)

        self.message_passing_layers = nn.ModuleList()
        for idx in range(num_layers):
            if idx == 0:
                self.message_passing_layers.append(AttentionLayer(hidden_dim, n_heads, 128*3, True))
            else:
                self.message_passing_layers.append(AttentionLayer(hidden_dim, n_heads, 128*3))

        self.decision_layer = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.embed_layer(x)
        for l in self.message_passing_layers:
            h = l(h, edge_index)
        h = self.decision_layer(h)
        return h


class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(self.adj_t_norm @ x)


class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )

class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(
            x, edge_index_t
        )


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
    ):
        super(GNN, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        output_dim = hidden_dim if jumping_knowledge else num_classes 
        
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha))
        
        if jumping_knowledge != False:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)
        else:
            self.lin = Linear(output_dim, num_classes)
        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.selu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge != False:
            x = self.jump(xs)
            x = self.lin(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.data import HeteroData
import torch_geometric as pyg

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = x
        h = self.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

class HeteroConv(nn.Module):
    def __init__(self, conv_list, aggr='sum'):
        super().__init__()
        self.conv_dict = nn.ModuleList(conv_list)
        self.aggr = aggr
        self.node_type = 'node1'

    def forward(self, x_dict, edge_index_dict):
        out_dict = {}
        for idx, edge_index in enumerate(edge_index_dict):
            conv = self.conv_dict[idx]
            if self.node_type not in x_dict or self.node_type not in x_dict:
                continue
            out = conv((x_dict[self.node_type], x_dict[self.node_type]), edge_index)
            if self.node_type not in out_dict:
                out_dict[self.node_type] = out
            else:
                if self.aggr == 'sum':
                    out_dict[self.node_type] += out
                elif self.aggr == 'mean':
                    out_dict[self.node_type] = (out_dict[self.node_type] + out) / 2
                elif self.aggr == 'max':
                    out_dict[self.node_type] = torch.max(out_dict[self.node_type], out)
        return out_dict

class RGCN4(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_layers=2, n_heads=2):
        super().__init__()
        self.rel_names = list(rel_names)
        self.embed_layer = MLP(in_feats, hid_feats, hid_feats)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_list = [pyg.nn.conv.GATConv(hid_feats, hid_feats // n_heads, heads=n_heads, add_self_loops=False).jittable('(Tensor, Tensor, OptTensor) -> Tensor') for rel in self.rel_names]
            self.gnn_layers.append(HeteroConv(conv_list, aggr='sum'))
        self.decision_layer = MLP(hid_feats, hid_feats, out_feats)

    def forward(self, edge_index_dict, inputs):
        x_dict = {'node1': self.embed_layer(inputs)}
        for gnn_layer in self.gnn_layers:
            x_dict = gnn_layer(x_dict, edge_index_dict)
            x_dict = {k: F.leaky_relu(v).flatten(1) for k, v in x_dict.items()}
            x_dict['node1'] += x_dict['node1']
        h = self.decision_layer(x_dict['node1'])
        return h



def get_model(args):
    
        return RGCN4(
            in_feats=args.num_features,
            hid_feats=args.hidden_dim,
            num_layers=args.num_layers,
            out_feats=args.num_classes,
            n_heads=8,
            rel_names = ['ss', 'st', 'ts', 'tt', 'pp']
        )
