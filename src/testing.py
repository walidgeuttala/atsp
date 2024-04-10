import numpy as np
import networkx as nx
import os 
import pickle

dir_path = '../../cleaned_data_n5900/'
output_path = '../../tsp_n5900/'
files = [dir_path+file for file in os.listdir(dir_path)]


edges_to_remove1 = [(node_i+64, node_i) for node_i in range(64)]
edges_to_remove2 = [(node_i, node_i+64) for node_i in range(64)]
edges_to_remove3 = [(node_i, node_j) for node_i in range(64) for node_j in range(64)]
edges_to_remove4 = [(node_i, node_j) for node_i in range(64, 128) for node_j in range(64, 128)]
idx = 0

for file in files:
    with open(file, 'rb') as f:
        G = pickle.load(f)

    G.remove_edges_from(edges_to_remove1)
    G.remove_edges_from(edges_to_remove2)
    G.remove_edges_from(edges_to_remove3)
    G.remove_edges_from(edges_to_remove4)
    with open(output_path+f'instance{idx}.pkl', 'wb') as file:
        pickle.dump(G, file)
    idx += 1