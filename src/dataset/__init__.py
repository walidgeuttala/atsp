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

def optimized_line_graph(g, args):
    n = g.number_of_nodes()
    m1 = n*(n-1)*(n-2)//2
    m2 = n*(n-1)//2
    if 'ss' in args.relation_types:
        ss = torch.empty((m1, 2), dtype=torch.int32)
    if 'st' in args.relation_types:
        if args.half_st:
            st = torch.empty((m1, 2), dtype=torch.int32)
        else:
            st = torch.empty((m1*2, 2), dtype=torch.int32)
    if 'tt' in args.relation_types:
        tt = torch.empty((m1, 2), dtype=torch.int32)
    if 'pp' in args.relation_types:
        pp = torch.empty((m2, 2), dtype=torch.int32)

    edge_id = {edge: idx for idx, edge in enumerate(g.edges())}
    idx = 0
    idx2 = 0
    for x in range(0, n):
        for y in range(0, n-1):
            if x != y:
                for z in range(y+1, n):
                    if x != z:
                        if 'ss' in args.relation_types:
                            ss[idx] = torch.tensor([edge_id[(x, y)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'st' in args.relation_types:
                            if args.half_st:
                                st[idx] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                            else:
                                st[idx*2] = torch.tensor([edge_id[(x, y)], edge_id[(z, x)]], dtype=torch.int32)
                                st[idx*2+1] = torch.tensor([edge_id[(y, x)], edge_id[(x, z)]], dtype=torch.int32)
                        if 'tt' in args.relation_types:
                            tt[idx] = torch.tensor([edge_id[(y, x)], edge_id[(z, x)]], dtype=torch.int32)
                        idx += 1
        if 'pp' in args.relation_types:
            for y in range(x+1, n):
                pp[idx2] = torch.tensor([edge_id[(x, y)], edge_id[(y, x)]], dtype=torch.int32)
                idx2 += 1
    edge_types = {}

    if 'ss' in args.relation_types:
        edge_types[('node1', 'ss', 'node1')] = (ss[:, 0], ss[:, 1])
    if 'st' in args.relation_types:
        edge_types[('node1', 'st', 'node1')] = (st[:, 0], st[:, 1])
    if 'tt' in args.relation_types:
        edge_types[('node1', 'tt', 'node1')] = (tt[:, 0], tt[:, 1])
    if 'pp' in args.relation_types:
        edge_types[('node1', 'pp', 'node1')] = (pp[:, 0], pp[:, 1])

    g2 = dgl.heterograph(edge_types)

    g2 = dgl.add_reverse_edges(g2)

    g2.ndata['e'] = torch.tensor(list(edge_id.keys()))

    return g2    


class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, scalers_file=None):
        if not isinstance(instances_file, pathlib.Path):
            instances_file = pathlib.Path(instances_file)
        self.root_dir = instances_file.parent
        self.graphs = []
        self.instances = sorted([line.strip() for line in open(instances_file)])
        if scalers_file is None:
            scalers_file = self.root_dir / 'scalers.pkl'
        scalers = pickle.load(open(scalers_file, 'rb'))
        if 'edges' in scalers: # for backward compatability
            self.scalers = scalers['edges']
        else:
            self.scalers = scalers

        # only works for homogenous datasets
        with open(self.root_dir / self.instances[0], 'rb') as file:
            G = pickle.load(file)
        lG = nx.line_graph(G)
        self.G = from_networkx(lG)
        self.mapping = dict(zip(range(lG.number_of_nodes()), lG.nodes()))


    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        with open(self.root_dir / self.instances[i], 'rb') as file:
            G = pickle.load(file)
     
        H = self.get_scaled_features(G)
        return H

    def get_scaled_features(self, G):
       
        # Step 3: Extract edge features and assign them to the PyG Data object
        weight = torch.tensor([G[self.mapping[u][0]][self.mapping[u][1]]['weight'] for u in range(self.G.num_nodes)], dtype=torch.float32)
        regret = torch.tensor([G[self.mapping[u][0]][self.mapping[u][1]]['regret'] for u in range(self.G.num_nodes)], dtype=torch.float32)
             
        H = self.G.clone()
        H.x = torch.tensor(self.scalers['weight'].transform(weight.view(-1, 1)), dtype=torch.float32)
        H.y = torch.tensor(self.scalers['regret'].transform(regret.view(-1, 1)), dtype=torch.float32)

        return H
