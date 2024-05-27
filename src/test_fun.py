import copy
import pathlib
import pickle

import networkx as nx
import numpy as np
import torch
import torch.utils.data
import pickle
import utils 
from torch_geometric.utils import from_networkx, add_remaining_self_loops
from torch.utils.data import Dataset


import copy
import pathlib
import pickle
import networkx as nx
import numpy as np
import torch
import torch.utils.data
import utils 
from torch_geometric.utils import from_networkx, add_remaining_self_loops
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import torch_geometric as pyg
import dgl
def set_features(G):
    for e in G.edges:
        G.edges[e]['features'] = np.array([
            G.edges[e]['weight'],
        ], dtype=np.float32)

def set_labels(G):
    optimal_cost = utils.optimal_cost(G)
    if optimal_cost == 0:
        optimal_cost = 1e-6
    if optimal_cost < 0:
        value = -1.
    else:
        value = 1.
    for e in G.edges:
        regret = 0.

        if not G.edges[e]['in_solution']: 

            tour = utils.fixed_edge_tour(G, e)
            cost = utils.tour_cost(G, tour)
            regret = (cost - optimal_cost) / optimal_cost * value
            
        G.edges[e]['regret'] = regret

# def set_labels2(G):
#     optimal_cost = optimal_cost(G)
#     for e in G.edges:
#         regret = 0.
#         G.edges[e]['regret'] = regret

# def string_graph(G1):
#     G2 = nx.Graph()
#     num_nodes = G1.number_of_edges()
#     nodes = range(0, num_nodes*(num_nodes+1)/2)
#     G2.add_nodes_from(nodes)
#     for edge in G1.edges():
#         s, t = edge
#         for neighbor in range(0, num_nodes):
#             if neighbor == t:
#                 pass
#             elif neighbor == s:
#                 pass
#             else:
#                 pass

import dgl

def directed_string_graph(G1):
    n = G1.number_of_nodes()
    m = n*(n-1)

    i, j = 0, 1
    ss = []
    st = []
    ts = []
    tt = []
    pp = []
    edge_id = dict()
    
    for idx, edge in enumerate(G1.edges()):
        edge_id[edge] = idx

    set_list = set()
    for idx in range(m):
        # parallel
        if (i, j, j, i) not in set_list:
            set_list.add((i, j, j, i))
            set_list.add((j, i, i, j))
            pp.append((edge_id[(i, j)], edge_id[(j, i)]))
        # src to src
        for v in range(n):
            if v != i and v != j and (i, j, i, v) not in set_list:
                set_list.add((i, j, i, v))
                set_list.add((i, v, i, j))
                ss.append((edge_id[(i, j)], edge_id[(i, v)]))
        # src to target
        for v in range(n):
            if v != i and v != j and (i, j, v, i) not in set_list:
                set_list.add((i, j, v, i))
                set_list.add((v, i, i, j))
                st.append((edge_id[(i, j)], edge_id[(v, i)]))
        # target to src
        for v in range(n):
            if v != i and v != j and (i, j, j, v) not in set_list:
                set_list.add((i, j, j, v))
                set_list.add((j, v, i, j))
                ts.append((edge_id[(i, j)], edge_id[(j, v)]))
        # target to target
        for v in range(n):
            if v != i and v != j and (i, j, v, j) not in set_list:
                set_list.add((i, j, v, j))
                set_list.add((v, j, i, j))
                tt.append((edge_id[(i, j)], edge_id[(v, j)]))
        
        j += 1
        if i == j:
            j += 1
        if j == n:
            j = 0
            i += 1

    edge_types = {('node1', 'ss', 'node1'): ss,
              ('node1', 'st', 'node1'): st,
              ('node1', 'ts', 'node1'): ts,
               ('node1', 'tt', 'node1'): tt,
               ('node1', 'pp', 'node1'): pp}
    
    G2 = dgl.heterograph(edge_types)
    G2 = dgl.add_reverse_edges(G2)
    
    #G2.ndata['e'] = torch.tensor(list(edge_id.keys()))
    # G2.ndata['e'] = torch.tensor(list(edge_id.keys())).clone()
    return from_dgl(G2)

from torch_geometric.data import HeteroData

def from_dgl(g):
    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")
    data = HeteroData()

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = torch.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    return data

class TSPDataset(Dataset):
    def __init__(self, instances_file, scalers_file=None):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)
        self.root_dir = instances_file.parent
        self.graphs = []
        self.instances = sorted([line.strip() for line in open(instances_file)])
        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        scalers = pickle.load(open(scalers_file, 'rb'))
        if 'edges' in scalers:  # for backward compatibility
            self.scalers = scalers['edges']
        else:
            self.scalers = scalers

        # only works for homogeneous datasets
        with open(self.root_dir / self.instances[0], 'rb') as file:
            G = pickle.load(file)
        # this takes a networkX graph and returns a line graph, undirected with 6 types of edges and 1 type of node as HeteroData
        # no need to define it
        self.G = directed_string_graph(G)
        # adding the self-loop to the graph
        for edge_type in self.G.edge_types:
            self.G[edge_type].edge_index = add_remaining_self_loops(self.G[edge_type].edge_index)[0]

        self.edge_id = self.get_edge_mapping(G)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        with open(self.root_dir / self.instances[i], 'rb') as file:
            G = pickle.load(file)
     
        H = self.get_scaled_features(G)
        return H

    def get_edge_mapping(self, G1):
        edge_id = {edge: idx for idx, edge in enumerate(G1.edges())}
        return edge_id

    def get_scaled_features(self, G):
        # Extract features from the original graph G
        features = np.zeros((len(self.edge_id), 3))  # Assuming 3 features: weight, regret, in_solution
        for edge, idx in self.edge_id.items():
            features[idx, 0] = G.edges[edge]['weight']
            features[idx, 1] = G.edges[edge]['regret']
            features[idx, 2] = G.edges[edge]['in_solution']

        # Scale the features
        weight = features[:, 0].reshape(-1, 1)
        regret = features[:, 1].reshape(-1, 1)
        in_solution = features[:, 2].reshape(-1, 1)

        weight_scaled = self.scalers['weight'].transform(weight)
        regret_scaled = self.scalers['regret'].transform(regret)

        # Update the transformed graph with scaled features
        H = copy.deepcopy(self.G)
        H['node1'].x = torch.tensor(weight_scaled, dtype=torch.float32)
        H['node1'].y = torch.tensor(regret_scaled, dtype=torch.float32)
        H['node1'].in_solution = torch.tensor(in_solution, dtype=torch.float32)

        return H
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

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

class CustomGATConv(GATConv):
    def jittable(self, input_type: str):
        return torch.jit.script(self)

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

from torch_geometric.loader import DataLoader

dataset = TSPDataset(instances_file='../../atsp_n5900/val.txt', scalers_file='../../atsp_n5900/scalers.pkl')
train_loader = DataLoader(dataset, batch_size=32)
model = RGCN4(in_feats=1, hid_feats=32, out_feats=1, rel_names=['ss', 'st', 'ts', 'tt', 'pp'])
device = "cuda"
model = model.to(device)

for batch_i, batch in enumerate(train_loader):
    batch = batch.to(device)
    x = batch['node1'].x
    y = batch['node1'].y
    y_pred = model(list(batch.edge_index_dict.values()), x)
    print(y_pred.shape)
    
    break


print(dataset[0].edge_index_dict)
print(dataset[0]['node1'].x)
traced_model = torch.jit.trace(model, (list(batch.edge_index_dict.values()), x))

model_scripted = torch.jit.script(traced_model)
model_scripted.save('walid.pt')
