import networkx as nx
import pickle

# Read the graph from pickle file
with open('../../tsp_n5900/instance0.pkl', 'rb') as f:
    G = pickle.load(f)

# Print the adjacency matrix
print(G.number_of_edges())
edge_weight, _ = nx.attr_matrix(G, 'weight')
print("Adjacency matrix:")
print(edge_weight)

# Save the adjacency matrix to a file
with open('sample.txt', 'w') as f:
    for row in edge_weight:
        f.write(' '.join(map(str, row)) + '\n')
