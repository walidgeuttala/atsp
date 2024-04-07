#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import multiprocessing as mp
import pathlib
import uuid

import networkx as nx
import numpy as np

from utils import utils
import dataset 

import linecache

def prepare_instance(G):
    dataset.set_features(G)
    dataset.set_labels(G)
    return G

def get_solved_instances_2D(n_nodes, n_instances):
    for _ in range(n_instances):
        G = nx.Graph()

        coords = np.random.random((n_nodes, 2))
        for n, p in enumerate(coords):
            G.add_node(n, pos=p)

        for i, j in itertools.combinations(G.nodes, 2):
            w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos'])
            G.add_edge(i, j, weight=w)

        opt_solution = utils.optimal_tour(G)
        in_solution = utils.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G


def get_solved_instances_atsp(n_nodes, n_instances, all_instances):
    #all_instances = './tsplib95_10000_instances_64_node/all_instances_lower_triangle_tour_cost.txt'
   
    for i in range(n_instances):
        line = linecache.getline(all_instances, i+1).strip()
        G = nx.Graph()
        adj, opt_solution, opt_cost = line.split(',')
        adj = adj.split(' ')

        G.add_nodes_from(range(n_nodes))
        opt_solution = [int(x) for x in opt_solution.split()]
       
        # Add the edges for the DiGraph and be sure that does not have self loops in the node
        for j in range(n_nodes):
            for k in range(n_nodes):
                w = float(adj[j*n_nodes+k])
                if j != k:
                    G.add_edge(j, k, weight=w)
            
        in_solution = utils.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset.')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('n_nodes', type=int)
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_dir', type=pathlib.Path)
    parser.add_argument('atsp', type=bool, default=True)
    # if arg atsp is True, we use this paramters for the transformation
    parser.add_argument('atsp', type=bool, default=True)

    args = parser.parse_args()

    if args.output_dir.exists():
        raise Exception(f'Output directory {args.output_dir} exists.')
    else:
        args.output_dir.mkdir()

    pool = mp.Pool(processes=None)
    instance_gen = get_solved_instances_atsp(args.n_nodes, args.n_samples, args.input_file)
    for G in pool.imap_unordered(prepare_instance, instance_gen):
        nx.write_gpickle(G, args.output_dir / f'{uuid.uuid4().hex}.pkl')
    pool.close()
    pool.join()



