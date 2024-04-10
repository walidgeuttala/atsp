import argparse
import os
import pickle
import networkx as nx
import pathlib


def tsp_to_atsp_instance(G1):
    num_nodes = G1.number_of_nodes() // 2
    G2 = nx.DiGraph()
    G2.add_nodes_from(range(num_nodes))
    G2.add_edges_from([(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v])

    first_edge = next(G1.edges())

    # Get the attribute names of the first edge
    attribute_names = G1[first_edge[0]][first_edge[1]].keys()
    attribute_names_list = list(attribute_names)
    for attribute_name in attribute_names_list:
        attribute, _ = nx.attr_matrix(G1, attribute_name)
        attribute = attribute[num_nodes:, :num_nodes]
        for u, v in G2.edges():
            G2[u][v][attribute_name] = attribute[u, v]
    
    return G2

def tsp_to_atsp(args):
    instances = list(args.input_dir.glob('instance*.pkl'))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for idx, instance in enumerate(instances):
        print(idx, end= " ")
        output_name = args.output_dir / f'instance{idx}.pkl'
        with open(instance, 'rb') as file:
            G1 = pickle.load(file)
        num_nodes = G1.number_of_nodes() // 2
        G2 = nx.DiGraph()
        G2.add_nodes_from(range(num_nodes))
        G2.add_edges_from([(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v])
        # Get the weight
        weight, _ = nx.attr_matrix(G1, 'weight')
        weight = weight[64:, :64]
        # Get the regret
        regret, _ = nx.attr_matrix(G1, 'regret')
        regret = regret[64:, :64]
        # Get the solution
        in_solution, _ = nx.attr_matrix(G1, 'in_solution')
        in_solution = in_solution[64:, :64]
        for u, v in G2.edges():
            G2[u][v]['weight'] = weight[u, v]
            G2[u][v]['regret'] = regret[u, v]
            G2[u][v]['in_solution'] = in_solution[u, v]
        with open(output_name, 'wb') as file:
            pickle.dump(G2, file)

def atsp_to_tsp(args):
    instances = list(args.input_dir.glob('instance*.pkl'))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for idx, instance in enumerate(instances):
        output_name = args.output_dir / f'instance{idx}.pkl'
        with open(instance, 'rb') as file:
            G1 = pickle.load(file)
        
        num_nodes = G1.number_of_nodes()
        print(num_nodes)
        G2 = nx.Graph()
        G2.add_nodes_from(range(num_nodes*2))
        G2.add_edges_from([(u+64, v) for u in range(num_nodes) for v in range(num_nodes) if u != v])
        # Get the weight
        weight, _ = nx.attr_matrix(G1, 'weight')
        # Get the regret
        regret, _ = nx.attr_matrix(G1, 'regret')
        # Get the regret
        in_solution, _ = nx.attr_matrix(G1, 'in_solution')
        # [INF][A.T]
        # [A  ][INF]
        for u, v in G2.edges():
            G2[u][v]['weight'] = weight[u%num_nodes, v%num_nodes]
            G2[u][v]['regret'] = regret[u%num_nodes, v%num_nodes]
            G2[u][v]['in_solution'] = in_solution[v%num_nodes, u%num_nodes]
           

        with open(output_name, 'wb') as file:
            pickle.dump(G2, file)


# Reformulate an asymmetric TSP as a symmetric TSP: 
# "Jonker and Volgenant 1983"
# This is possible by doubling the number of nodes. For each city a dummy 
# node is added: (a, b, c) => (a, a', b, b', c, c')

# distance = "value"
# distance (for each pair of dummy nodes and pair of nodes is INF)
# distance (for each pair node and its dummy node is DIAG)
# ------------------------------------------------------------------------
#   |A'   |B'   |C'   |A    |B    |C    |
# A'|0    |INF  |INF  |DIAG |dBA  |dCA  |
# B'|INF  |0    |INF  |dAB  |DIAG |dCB  | 
# C'|INF  |INF  |0    |dAC  |dBC  |DIAG |
# A |DIAG |dAB  |dAC  |0    |INF  |INF  |
# B |dBA  |DIAG |dBC  |INF  |0    |INF  |
# C |dCA  |dCB  |DIAG |INF  |INF  |0    |
# 
# for the paper the DIAG = -INF, but you could have (INF = 0 and DIAG=-1e6) or (INF = 1e6 and DIAG = -INF)
#
# [INF][A.T]
# [A  ][INF]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset.')
    parser.add_argument('input_dir', type=pathlib.Path)
    parser.add_argument('output_dir', type=pathlib.Path)
    parser.add_argument('atsp_to_tsp', type=bool, default=True)
    args = parser.parse_args()

    args.atsp_to_tsp = True
    if args.atsp_to_tsp:
        atsp_to_tsp(args)
    else:
        tsp_to_atsp(args)
