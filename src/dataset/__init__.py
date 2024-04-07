import copy
import pathlib
import pickle

import networkx as nx
import numpy as np
import torch
import torch.utils.data
import pickle
import utils 
from torch_geometric.utils import from_networkx


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
        original_weight = weight.clone()
        original_regret = regret.clone()   
             
        H = self.G.clone()
        H.x = torch.tensor(self.scalers['weight'].transform(weight.view(-1, 1)), dtype=torch.float32)
        H.y = torch.tensor(self.scalers['regret'].transform(regret.view(-1, 1)), dtype=torch.float32)
        H.original_weight = original_weight
        H.original_regret = original_regret

        return H
