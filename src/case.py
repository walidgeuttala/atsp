import networkx as nx
import torch
import torch_geometric
from torch_geometric.utils import from_networkx

# Step 1: Create a NetworkX graph
G_nx = nx.DiGraph()
G_nx.add_nodes_from([0, 1, 2, 3])
edges = [(0, 1), (1, 2), (2, 0)]
for u, v in edges:
    G_nx.add_edge(u, v, weight=5, regret=10)

lG = nx.line_graph(G_nx)
mapping = dict(zip(range(lG.number_of_nodes()), lG.nodes()))

G_pyg = from_networkx(lG)
print(mapping[0][0])
print(G_nx[0][1]['weight'])
print(torch.tensor([G_nx[mapping[u][0]][mapping[u][1]]['weight'] for u in range(G_pyg.num_nodes)], dtype=torch.float32))


# print(lG)
# print(G_nx)
# for node in lG.nodes():
#     print(node)
# # Step 2: Transform the NetworkX graph into a PyG Data object
# G_pyg = from_networkx(lG)
# print(G_pyg.x)

# # Step 3: Extract edge features and assign them to the PyG Data object
# weight = torch.tensor([G_nx[u.item()][v.item()]['weight'] for u, v in G_pyg.edge_index.T], dtype=torch.float32)
# print(weight)
# regret = [G_nx[u.item()][v.item()]['regret'] for u, v in G_pyg.edge_index.T]
# G = from_networkx(G_nx)
# G = G.line_graph()
# print(G.regret)
# print(G.weight)
# # G_pyg.weight = torch.tensor(weight, dtype=torch.float32)
# # G_pyg.regret = torch.tensor(regret, dtype=torch.float32)

# #  # Step 3: Extract edge features and assign them to the PyG Data object
# #         weight = torch.tensor([G[u.item()][v.item()]['weight'] for u, v in self.G.edge_index.T], dtype=torch.float32)
# #         regret = torch.tensor([G[u.item()][v.item()]['regret'] for u, v in self.G.edge_index.T], dtype=torch.float32)
# #         original_weight = weight.clone()
# #         original_regret = regret.clone()        

# #         self.G.weight = self.scalers['weight'].transform(weight.view(-1, 1)).view(-1)
# #         self.G.regret = self.scalers['regret'].transform(regret.view(-1, 1)).view(-1)
# #         self.G.original_weight = original_weight
# #         self.G.original_regret = original_regret
