import networkx as nx
import torch
from torch_geometric.data import Data
import torch_geometric
# Step 1: Create a NetworkX graph
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3])
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
for u, v in edges:
    G.add_edge(u, v, weight=0, regret=0)

# Step 2: Convert the original graph to its line graph
G = nx.line_graph(G)
print(G)
data = torch_geometric.utils.convert.from_networkx(G)
print(data)