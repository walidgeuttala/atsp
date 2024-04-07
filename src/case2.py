
import networkx as nx
import dgl
import numpy as np
import copy
import torch
from torch_geometric.data import Data

# Step 1: Create a NetworkX graph
G_nx = nx.Graph()
G_nx.add_nodes_from([0, 1, 2])
edges = [(0, 1), (1, 2), (2, 0)]
for u, v in edges:
    G_nx.add_edge(u, v, weight=5, regret=10)
lG = nx.line_graph(G_nx)
for n in lG.nodes:
    lG.nodes[n]['e'] = n
# Step 2: Transform the NetworkX graph into a DGL graph
G_dgl = dgl.from_networkx(lG, node_attrs=['e'])


weight = []
regret = []
for i in range(G_dgl.num_nodes()):
    e = tuple(G_dgl.ndata['e'][i].numpy())
    
    weight.append(G_nx.edges[e]['weight'])
    regret.append(G_nx.edges[e]['regret'])

weight = np.vstack(weight)
regret = np.vstack(regret)
H = copy.deepcopy(G_dgl)
H.ndata['weight'] = torch.tensor(weight, dtype=torch.float32)
H.ndata['regret'] = torch.tensor(regret, dtype=torch.float32)
print(torch.tensor(weight, dtype=torch.float32))
print(torch.tensor(regret, dtype=torch.float32))

src, dst = self.G.edges()
edge_index = torch.stack([src, dst], dim=0)
data = Data(x=H.ndata['weight'], y=H.ndata['regret'], edge_index=edge_index)
data.original_regret = H.ndata['original_regret']
