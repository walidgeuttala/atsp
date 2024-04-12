import networkx as nx
import pickle
import numpy as np
# Read the graph from pickle file
with open('../../tsp_n5900/instance0.pkl', 'rb') as f:
    G = pickle.load(f)
with open('../../atsp_n5900/instance0.pkl', 'rb') as f:
    G2 = pickle.load(f)

# Print the adjacency matrix

edge_weight, _ = nx.attr_matrix(G, 'weight')
edge_weight2, _ = nx.attr_matrix(G2, 'weight')
print("Adjacency matrix:")

print(edge_weight.T[64, :64])
print(edge_weight2[0, :])
if np.array_equal(edge_weight[64:, :64].T,edge_weight2):
    print('yes')
else:
    print('no')
# Save the adjacency matrix to a file
with open('sample.txt', 'w') as f:
    for row in edge_weight:
        f.write(' '.join(map(str, row)) + '\n')
