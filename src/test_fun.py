import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.data import HeteroData

def directed_string_graph(G1):
    n = G1.number_of_nodes()
    m = n * (n - 1)

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

    edge_types = {
        'ss': ss,
        'st': st,
        'ts': ts,
        'tt': tt,
        'pp': pp
    }

    data = HeteroData()

    for etype, edge_list in edge_types.items():
        if edge_list:
            src, dst = zip(*edge_list)
            data['node1', etype, 'node1'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = pyg.transforms.ToUndirected(data)

    edge_id_tensor = torch.tensor(list(edge_id.keys()), dtype=torch.long)
    data['node1'].x = edge_id_tensor.unsqueeze(-1)  # Add node features

    return data

# Example usage:
G1 = nx.complete_graph(4, nx.DiGraph())
G2 = directed_string_graph(G1)
print(G2)
for etype in ['ss', 'st', 'ts', 'tt', 'pp']:
    print(f"{etype}: ", G2['node1', etype, 'node1'].num_edges)



G1 = nx.complete_graph(4, nx.DiGraph())
G2, edge_id  = directed_string_graph(G1)
print()
print(G2)