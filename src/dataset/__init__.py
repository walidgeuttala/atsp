import copy
import pathlib
import pickle

import numpy as np
import torch
import torch.utils.data
import pickle
import utils 
from torch_geometric.utils import from_networkx, add_remaining_self_loops
from torch.utils.data import Dataset
from torch_geometric.data import Data

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

def optimized_line_graph(g):
    n = g.number_of_nodes()
    m1 = n*(n-1)*(n-2)//2
    m2 = n*(n-1)//2
    relation_types = 'ss tt pp st'
    if 'ss' in relation_types:
        ss = torch.empty((m1, 2), dtype=torch.int64)
    else:
        ss = torch.empty((0, 2), dtype=torch.int64)
    if 'st' in relation_types:
        st = torch.empty((m1*2, 2), dtype=torch.int64)
    else:
        st = torch.empty((0, 2), dtype=torch.int64)

    if 'tt' in relation_types:
        tt = torch.empty((m1, 2), dtype=torch.int64)
    else:
        tt = torch.empty((0, 2), dtype=torch.int64)

    if 'pp' in relation_types:
        pp = torch.empty((m2, 2), dtype=torch.int64)
    else:
        pp = torch.empty((0, 2), dtype=torch.int64)


    edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
    idx = 0
    idx2 = 0
    for x in range(0, n):
        for y in range(0, n-1):
            if x != y:
                for z in range(y+1, n):
                    if x != z:
                        if 'ss' in relation_types:
                            ss[idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int64)
                        if 'st' in relation_types:
                            st[idx*2] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int64)
                            st[idx*2+1] = torch.tensor([edge_id[(y, x)], edge_id[(x, z)]], dtype=torch.int64)
                        if 'tt' in relation_types:
                            tt[idx] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int64)
                        idx += 1
        if 'pp' in relation_types:
            for y in range(x+1, n):
                pp[idx2] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int64)
                idx2 += 1
    edge_types = {}

    # if 'ss' in relation_types:
    #     edge_types[('node1', 'ss', 'node1')] = (ss[:, 0], ss[:, 1])
    # if 'st' in relation_types:
    #     edge_types[('node1', 'st', 'node1')] = (st[:, 0], st[:, 1])
    # if 'tt' in relation_types:
    #     edge_types[('node1', 'tt', 'node1')] = (tt[:, 0], tt[:, 1])
    # if 'pp' in relation_types:
    #     edge_types[('node1', 'pp', 'node1')] = (pp[:, 0], pp[:, 1])

  
    # g2 = dgl.heterograph(edge_types)
    # g2 = dgl.add_reverse_edges(g2)
    merged_edge_pairs = torch.cat([ss, st, tt, pp], dim=0)


    return Data(edge_index=merged_edge_pairs, edge_attr=torch.tensor(list(edge_id.keys())))    



def from_dgl(g):
    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")
    data = Data()

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
        self.G = optimized_line_graph(G)
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
        H.x = torch.tensor(weight_scaled, dtype=torch.float32)
        H.y = torch.tensor(regret_scaled, dtype=torch.float32)
        H.in_solution = torch.tensor(in_solution, dtype=torch.float32)

        return H