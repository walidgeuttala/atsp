import tqdm.auto as tqdm
import argparse
import datetime
import json
import pathlib
import time
import uuid

import networkx as nx
import numpy as np
import pandas as pd
import torch

from model import get_model
from train import train_parse_args, test
import algorithms
import utils
from dataset import TSPDataset

#import tqdm.auto as tqdm
def tour_cost2(tour, weight):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += weight[e]
    return c

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
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    params_train = json.load(open(args.model_path.parent / 'params.json'))
    args_train = train_parse_args()
    for key, value in params_train.items():
        setattr(args_train, key, value)

    test_data = TSPDataset(args.data_path)

    if 'regret_pred' in args.guides:
        device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
        model = get_model(args).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    pbar = tqdm.tqdm(test_data.instances)
    init_gaps = []
    final_gaps = []
    search_progress = []
    for instance in pbar:
        G = nx.read_gpickle(test_data.root_dir / instance)
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
            
            for idx in range(num_nodes*2):
                G.edges[test_data.mapping[idx]]['regret_pred'] = np.maximum(regret_pred[idx].item(), 0.)

            init_tour = algorithms.nearest_neighbor(G, 0, weight='regret_pred')
            

        init_cost = utils.tour_cost(G, init_tour)
        best_tour, best_cost, search_progress_i, cnt_ans = algorithms.guided_local_search(G, init_tour, init_cost,
                                                                                 t + args.time_limit, weight='weight',
                                                                                 guides=args.guides,
                                                                                 perturbation_moves=args.perturbation_moves,
                                                                                 first_improvement=False, value=0)
        for row in search_progress_i:
            row.update({
                'instance': instance,
                'opt_cost': opt_cost
            })
        search_progress.append(row)
        # print('tour : ',best_tour)
        # edge_weight, _ = nx.attr_matrix(G, 'weight')
        # print('orignal cost: ', tour_cost2(best_tour, edge_weight)+value)
        # print('init_tour cost: ', tour_cost2(init_tour, edge_weight)+value)
        # print(best_cost)
     
        if init_cost != best_cost:
            print('opt : ',opt_cost)
            print('init : ',init_cost)
            print('best : ',best_cost)
        # print(init_tour)
        # orignal_tour = [x for idx, x in enumerate(init_tour) if idx % 2 == 0]
        # print(orignal_tour)
        # print(opt_cost)
        # print(init_cost)
        edge_weight, _ = nx.attr_matrix(G, 'weight')
        # orignal_weights = edge_weight[ 64:, :64]
        # print(orignal_weights)
        # print('orignal cost: ', tour_cost2(orignal_tour, orignal_weights))
        regret, _ = nx.attr_matrix(G, 'regret')
        regret_pred, _ = nx.attr_matrix(G, 'regret_pred')
        # if cnt == 0:
        #     print(edge_weight,flush=True)
        #     print(regret,flush=True)
        #     print(regret_pred,flush=True)
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
        
        init_gap = (init_cost / opt_cost - 1) * 100
        final_gap = (best_cost / opt_cost - 1) * 100
        
        init_gaps.append(init_gap)
        final_gaps.append(final_gaps)
        print('Avg Gap init: {:.4f}'.format(np.mean(init_gaps)))
        print('Avg Gap best: {:.4f}'.format(np.mean(final_gaps)))        
        
        

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
