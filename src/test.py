import tqdm.auto as tqdm
import argparse
import datetime
import json
import pathlib
import time
import uuid
import pickle 

import networkx as nx
import numpy as np
import pandas as pd
import torch

from model import get_model
import algorithms
import utils
from dataset import TSPDataset

#import tqdm.auto as tqdm
def tour_cost2(tour, weight):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += weight[e]
    return c

def train_parse_args(values):
    parser = argparse.ArgumentParser("ATSP Graph Neural Network")

    ### Dataset Args
    parser.add_argument("--dataset", type=str, help="Name of dataset", default="atsp")
    parser.add_argument("--dataset_directory", type=pathlib.Path, help="Directory to save datasets", default="../../atsp_n5900")
    parser.add_argument("--tb_dir", type=pathlib.Path, help="Directory to save checkpoints", default="../../checkpoint")

    ### Preprocessing Args
    parser.add_argument("--undirected", action="store_true", help="Whether to use undirected version of graph")
    parser.add_argument("--self_loops", action="store_true", help="Whether to add self-loops to the graph")
    # parser.add_argument("--transpose", action="store_true", help="Whether to use transpose of the graph")

    ### Model Args
    parser.add_argument("--model", type=str, help="Model type", default="gnn")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=2)
    parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.0)
    parser.add_argument("--alpha", type=float, help="Direction convex combination params", default=0.5)
    parser.add_argument("--learn_alpha", action="store_true")
    parser.add_argument("--conv_type", type=str, help="DirGNN Model", default="dir-gcn")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--jk", type=str, choices=["max", "cat", None], default="cat")
    parser.add_argument('--num_features', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--target', type=str, default='regret')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')

    ### Training Args
    parser.add_argument("--lr_init", type=float, help="Learning Rate", default=0.001)
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0001)
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument("--patience", type=int, help="Patience for early stopping", default=10)
    parser.add_argument("--num_runs", type=int, help="Max number of runs", default=1)
    parser.add_argument('--checkpoint_freq', type=int, default=20, help='Checkpoint frequency')

    ### System Args
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--num_workers", type=int, help="Num of workers for the dataloader", default=16)

    
    args = parser.parse_args(values)


    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('data_path', type=pathlib.Path)
    parser.add_argument('model_path', type=pathlib.Path)
    parser.add_argument('run_dir', type=pathlib.Path)
    parser.add_argument('guides', type=str, nargs='+')
    parser.add_argument('output_path', type=pathlib.Path)
    parser.add_argument('--time_limit', type=float, default=10.)
    parser.add_argument('--perturbation_moves', type=int, default=20)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--num_features', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--save_prediction', type=bool, default=True)
    parser.add_argument('--tsp_as_atsp', type=bool, default=False)
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    params_train = json.load(open(args.model_path.parent / 'params.json'))
    args_train = train_parse_args([])
    for key, value in params_train.items():
        setattr(args_train, key, value)

    test_data = TSPDataset(args.data_path)

    if 'regret_pred' in args.guides:
        device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        model = get_model(args_train).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    pbar = tqdm.tqdm(test_data.instances)
    init_gaps = []
    final_gaps = []
    search_progress = []
    cnt = 0
    for instance in pbar:
        with open(test_data.root_dir / instance, 'rb') as file:
            G = pickle.load(file)
        num_nodes = G.number_of_nodes()
        opt_cost = utils.optimal_cost(G, weight='weight')

        t = time.time()
        search_progress.append({
            'instance': instance,
            'time': t,
            'opt_cost': opt_cost
        })

        if 'regret_pred' in args.guides:
            H = test_data.get_scaled_features(G).to(device)
            
            with torch.no_grad():
                y_pred = model(H.x, H.edge_index)
            regret_pred = test_data.scalers['regret'].inverse_transform(y_pred.cpu().numpy())
            for idx in range(len(G.edges())):
                G[test_data.mapping[idx][0]][test_data.mapping[idx][1]]['regret_pred'] = np.maximum(regret_pred[idx].item(), 0.)
            
            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')
            
        edge_weight, _ = nx.attr_matrix(G, 'weight')
        if args.tsp_as_atsp:
            for i in range(64):
                for j in range(i+1,64):
                    if i != j:
                        G.add_edge(i, j, weight=float(1e6), regret_pred=float(100), regret=float(100))
            for i in range(64,128):
                for j in range(64+i+1,128):
                    if i != j:
                        G.add_edge(i, j, weight=float(1e6), regret_pred=float(100), regret=float(100))

        init_cost = utils.tour_cost(G, init_tour)
        best_tour, best_cost, search_progress_i, cnt_ans = algorithms.guided_local_search(G, init_tour, init_cost,
                                                                                 t + args.time_limit, weight='weight',
                                                                                 guides=args.guides,
                                                                                 perturbation_moves=args.perturbation_moves,
                                                                                 first_improvement=False)
        for row in search_progress_i:
            row.update({
                'instance': instance,
                'opt_cost': opt_cost
            })
        
        search_progress.append(row)
        if args.tsp_as_atsp:
            init_cost += num_nodes/2*1e6
            opt_cost += num_nodes/2*1e6
            best_cost += num_nodes/2*1e6
        edge_weight, _ = nx.attr_matrix(G, 'weight')
        regret, _ = nx.attr_matrix(G, 'regret')
        regret_pred, _ = nx.attr_matrix(G, 'regret_pred')
     
        with open(args.output_path / f"instance{cnt}.txt", "w") as f:
            # Save array1
            f.write("edge_weight:\n")
            np.savetxt(f, edge_weight, fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array2
            f.write("regret:\n")
            np.savetxt(f, regret, fmt="%.8f", delimiter=" ")
            f.write("\n")

            # Save array3
            f.write("regret_pred:\n")
            np.savetxt(f, regret_pred, fmt="%.8f", delimiter=" ")
            f.write("\n")

            f.write(f"opt_cost: {opt_cost}\n")
            f.write(f"num_iterations: {cnt_ans}\n")
            f.write(f"init_cost: {init_cost}\n")
            f.write(f"best_cost: {best_cost}\n")
        
        cnt += 1
        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (best_cost / opt_cost - 1) * 100
        
        init_gaps.append(init_gap)
        final_gaps.append(final_gap)
        
        pbar.set_postfix({
                'Avg Gap init:': '{:.4f}'.format(np.mean(init_gaps)),
                'Avg Gap best:': '{:.4f}'.format(np.mean(final_gaps)),
                'optimal': f'{opt_cost}',
                'init': f'{init_cost}',
                'best': f'{best_cost}'
            })
        

    search_progress_df = pd.DataFrame.from_records(search_progress)
    search_progress_df['best_cost'] = search_progress_df.groupby('instance')['cost'].cummin()
    search_progress_df['gap'] = (search_progress_df['best_cost'] / search_progress_df['opt_cost'] - 1) * 100
    search_progress_df['dt'] = search_progress_df['time'] - search_progress_df.groupby('instance')['time'].transform(
        'min')

    timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    run_name = f'{timestamp}_{uuid.uuid4().hex}.pkl'
    if not args.run_dir.exists():
        args.run_dir.mkdir()
    search_progress_df.to_pickle(args.run_dir / run_name)
