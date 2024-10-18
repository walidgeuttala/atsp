#import concorde.tsp as concorde
#import lkh
import networkx as nx
import numpy as np
#import tsplib95
#from matplotlib import colors
import linecache
import torch
import pickle

def nearest_neighbor(G, depot, weight='weight'):
    tour = [depot]
    while len(tour) < len(G.nodes):
        i = tour[-1]
        neighbours = [(j, G.edges[(i, j)][weight]) for j in G.neighbors(i) if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(depot)
    return tour

def atsp_results(model, args, val_data):
    result2 = dict()
    keys = ['avg_corr', 'avg_corr_cosin', 'avg_init_cost', 'avg_opt_cost', 'avg_gap']
    for key in keys:
        result2.setdefault(key, 0.)
    for idx in range(30):
        with open(args.dataset_directory / val_data.instances[0], 'rb') as file:
            G = pickle.load(file)
        H = val_data.get_scaled_features(G).to(args.device)
        x = H.x
        with torch.no_grad():
            y_pred = model(list(H.edge_index_dict.values()), x)
        regret_pred = val_data.scalers['regret'].inverse_transform(y_pred.cpu().numpy())
        for edge, idx in val_data.edge_id.items():
            G.edges[edge]['regret_pred'] = np.maximum(regret_pred[idx].item(), 0.)
                
        opt_cost = optimal_cost(G, weight='weight')
        init_tour = nearest_neighbor(G, 0, weight='regret_pred')
        init_cost = tour_cost(G, init_tour)
        result2['avg_corr'] += correlation_matrix(y_pred.cpu(), H.y.cpu())
        result2['avg_corr_cosin'] += cosine_similarity(y_pred.cpu().flatten(), H.y.cpu().flatten())
        result2['avg_init_cost'] += init_cost
        result2['avg_opt_cost'] += opt_cost
        result2['avg_gap'] += (init_cost / opt_cost - 1) * 100

    return result2

def correlation_matrix(tensor1, tensor2):
    
    # Flatten tensors into 1D arrays
    flat_tensor1 = tensor1.flatten().numpy()
    flat_tensor2 = tensor2.flatten().numpy()

    # Concatenate flattened tensors along the second axis

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(flat_tensor1, flat_tensor2)[0, 1]
    
    return corr_matrix

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / (norm_A * norm_B)
    return similarity

def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges or tuple(reversed(e)) in tour_edges
    return in_tour


def tour_cost(G, tour, weight='weight'):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += G.edges[e][weight]
    return c

def add_diag(t1, num_nodes = 64):
    t2 = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    cnt = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            t2[i][j] = t1[cnt]
            cnt += 1
    return t2

def tour_cost2(tour, weight):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += weight[e]
    return c


def is_equivalent_tour(tour_a, tour_b):
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False


def is_valid_tour(G, tour):
    if tour[0] != 0:
        return False
    if tour[-1] != 0:
        return False
    for n in G.nodes:
        c = tour.count(n)
        if n == 0:
            if c != 2:
                return False
        elif c != 1:
            return False
    return True


# def optimal_tour(G, scale=1e3):
#     coords = scale * np.vstack([G.nodes[n]['pos'] for n in sorted(G.nodes)])
#     solver = concorde.TSPSolver.from_data(coords[:, 0], coords[:, 1], norm='EUC_2D')
#     solution = solver.solve()
#     tour = solution.tour.tolist() + [0]
#     return tour


def tranfer_tour(tour, x):
    result_list = []
    for num in tour:
        result_list.append(num)
        result_list.append(num + x)
    return result_list[:-1]

def as_symmetric(matrix, INF = 1e6):
    shape = len(matrix)
    mat = np.identity(shape) * - INF + matrix

    new_shape = shape * 2
    new_matrix = np.ones((new_shape, new_shape)) * INF
    np.fill_diagonal(new_matrix, 0)

    # insert new matrices
    new_matrix[shape:new_shape, :shape] = mat
    new_matrix[:shape, shape:new_shape] = mat.T
    # new cost matrix after transformation

    return new_matrix

def convert_adj_string(adjacency_matrix):
  ans = ''
  n = adjacency_matrix.shape[0]
  for i in range(n):
    # Iterate over columns up to the diagonal
      for j in range(n):
        ans += str(adjacency_matrix[i][j]) + " "
  return ans




def tranfer_tour(tour, x):
        result_list = []
        for num in tour:
            result_list.append(num)
            result_list.append(num + x)
        return result_list[:-1]

def append_text_to_file(filename, text):
    with open(filename, 'a') as file: file.write(text + '\n')


def atsp_to_tsp():
    value = 64e6
    for i in range(10):
        line = linecache.getline('../tsplib95_10000_instances_64_node/all_instances_adj_tour_cost.txt', i+2).strip()
        adj, opt_solution, cost = line.split(',')
        cost = float(cost)
        cost -= value
        adj = adj.split(' ')[:-1]
        opt_solution = [int(x) for x in opt_solution.split()]
        adj = np.array(adj, dtype=np.int32).reshape(64, 64)
        adj = gnngls.as_symmetric(adj)
        opt_solution = tranfer_tour(opt_solution, 64)
        instance_adj_tour_cost = gnngls.convert_adj_string(adj)+','+" ".join(map(str, opt_solution))+','+str(cost)
        append_text_to_file('../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt', instance_adj_tour_cost)

def adjacency_matrix_to_networkx(adj_matrix):
    return nx.Graph(np.triu(adj_matrix))

def optimal_cost(G, weight='weight'):
    c = 0
    for e in G.edges:
        if G.edges[e]['in_solution']:
            c += G.edges[e][weight]
    return c


def get_adj_matrix_string(G):
    # Get the lower triangular adjacency matrix with diagonal
    adj_matrix = nx.to_numpy_array(G).astype(int)
    n = adj_matrix.shape[0]
    ans = f'''NAME: TSP
    COMMENT: 64-city problem
    TYPE: TSP
    DIMENSION: {n}
    EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX
    EDGE_WEIGHT_SECTION: 
    '''
    for i in range(n):
        # Iterate over columns up to the diagonal
        for j in range(n):
            ans += str(adj_matrix[i][j]) + " "
        ans += "\n"
    # Add EOF
    # adj_matrix_string += "EOF"
    
    return ans.strip()


def fixed_edge_tour(G, e, lkh_path='../LKH-3.0.9/LKH'):
    string = get_adj_matrix_string(G)
    problem = tsplib95.loaders.parse(string)
    problem.fixed_edges = [[n + 1 for n in e]]

    solution = lkh.solve(lkh_path, problem=problem)
    tour = [n - 1 for n in solution[0]] + [0]
    return tour


def plot_edge_attribute(G, attr, ax, **kwargs):
    cmap_colors = np.zeros((100, 4))
    cmap_colors[:, 0] = 1
    cmap_colors[:, 3] = np.linspace(0, 1, 100)
    cmap = colors.ListedColormap(cmap_colors)

    pos = nx.get_node_attributes(G, 'pos')

    nx.draw(G, pos, edge_color=attr.values(), edge_cmap=cmap, ax=ax, **kwargs)
