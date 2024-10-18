import tqdm.auto as tqdm
import argparse
import logging
import datetime
import uuid
import json
import pathlib
import os 
import pickle
import numpy as np


import torch 
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import get_model
import dataset
import utils
import algorithms


def loss3(y_pred, y, args):
    num_edges = int(y.shape[0]//args.batch_size)
    y = y.view(args.batch_atsp_resultssize, num_edges) ** 2
    y = F.normalize(y, p=2, dim=1)
    
    y_pred = y_pred.view(args.batch_size, num_edges) ** 2
    y_pred = F.normalize(y_pred, p=2, dim=1)
    
    cos_similarities = F.cosine_similarity(y, y_pred, dim=1)

    return 1 - cos_similarities.mean()

def train(model, data_loader, criterion, optimizer, args):
    model.train()

    epoch_loss = 0
    cnt = 0
    for  batch in data_loader:
        batch = batch.to(args.device)
        x = batch.x
        y = batch.y
        optimizer.zero_grad()
        y_pred = model(batch.edge_index, x)
        loss = criterion(y_pred.squeeze(), y.type_as(y_pred).squeeze())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        cnt += batch.num_graphs

    epoch_loss /= cnt
    return epoch_loss

def test(model, data_loader, criterion, args):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        cnt = 0
        for  batch in data_loader:
            batch = batch.to(args.device)
            x = batch.x
            y = batch.y

            y_pred = model(batch.edge_index, x)
            loss = criterion(y_pred.squeeze(), y.type_as(y_pred).squeeze())
            epoch_loss += loss
            cnt += batch.num_graphs
        epoch_loss /= cnt
        return epoch_loss

def save(model, optimizer, epoch, train_loss, val_loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'val_loss': val_loss
    }, save_path)

def train_parse_args():
    parser = argparse.ArgumentParser("ATSP Graph Neural Network")

    ### Dataset Args
    parser.add_argument("--dataset", type=str, help="Name of dataset", default="atsp")
    parser.add_argument("--dataset_directory", type=pathlib.Path, help="Directory to save datasets", default="../../tsp_input/generated_insatnces_3000_size_50")
    parser.add_argument("--tb_dir", type=pathlib.Path, help="Directory to save checkpoints", default="../../checkpoint")

    ### Model Args
    parser.add_argument("--model", type=str, help="Model type", default="gnn")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=128)
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=4)
    parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.)
    parser.add_argument("--alpha", type=float, help="Direction convex combination params", default=0.5)
    parser.add_argument("--learn_alpha", action="store_true")
    parser.add_argument("--conv_type", type=str, help="DirGNN Model", default="dir-gcn")
    parser.add_argument("--gat_model", action="store_true", help="GAT with skip connetion",)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--jk", type=str, choices=["max", "cat", False], default=False)
    parser.add_argument('--num_features', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--target', type=str, default='regret')

    ### Training Args
    parser.add_argument("--lr_init", type=float, help="Learning Rate", default=0.0001)
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.95)
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument("--patience", type=int, help="Patience for early stopping", default=10)
    parser.add_argument("--num_runs", type=int, help="Max number of runs", default=1)
    parser.add_argument('--checkpoint_freq', type=int, default=5, help='Checkpoint frequency')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--batch_size', type=int, default=15, help='Batch size')

    ### System Args
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--num_workers", type=int, help="Num of workers for the dataloader", default=4)

    args = parser.parse_args()

    return args

def loss2(y, args):
    num_edges = int(y.shape[0]//args.batch_size ** 0.5)
    y = y.view(args.batch_size, num_edges, num_edges)

def load_checkpoint(model, optimizer, filepath):
    """Load model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['loss']
    val_loss = checkpoint['val_loss']
    
    return epoch, train_loss, val_loss

def run(args):
    torch.manual_seed(0)
    print(args)
    for epoch in range(args.num_runs):
         
        train_data = dataset.TSPDataset(f"{args.dataset_directory}/train.txt")
        val_data = dataset.TSPDataset(f"{args.dataset_directory}/val.txt")
        
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        model = get_model(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.weight_decay)
        print(f'device : {args.device}')

        if args.target == 'regret':
            criterion = torch.nn.MSELoss()

        timestamp = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = ""
        for i in range(1000):
            run_name = f'version{i}'
            log_dir = args.tb_dir / run_name
            if os.path.exists(log_dir):
                continue
            else:
                os.mkdir(log_dir)
                break
        writer = SummaryWriter(log_dir)

        # early stopping
        best_score = None
        counter = 0
        H = None
        x = None
        pbar = tqdm.trange(args.n_epochs)
        for epoch in pbar:
            epoch_loss = train(model, train_loader, criterion, optimizer, args)
            writer.add_scalar("Loss/train", epoch_loss, epoch)

            epoch_val_loss = test(model, val_loader, criterion, args)
            writer.add_scalar("Loss/validation", epoch_val_loss, epoch)
            

            result = utils.atsp_results(model, args, val_data)
            formatted_result = {key: f'{(value/30):.4f}' for key, value in result.items()}  # Format values to 4 decimal places
            formatted_result['train_loss'] = f"{epoch_loss:.4f}"
            formatted_result['val_loss'] = f"{epoch_val_loss:.4f}"
            formatted_result['epoch'] = f'{epoch:.4f}'
            pbar.set_postfix(**formatted_result)
            
            if args.checkpoint_freq is not None and epoch > 0 and epoch % args.checkpoint_freq == 0:
                checkpoint_name = f'checkpoint_{epoch}.pt'
                save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / checkpoint_name)

            if best_score is None or epoch_val_loss < best_score - args.min_delta:
                save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_best_val.pt')

                best_score = epoch_val_loss
                counter = 0
            else:
                counter += 1

            if counter >= args.patience:
                pbar.close()
                break

            lr_scheduler.step()

        writer.close()

        params = dict(vars(args))
        params['dataset_directory'] = str(params['dataset_directory'])
        params['tb_dir'] = str(params['tb_dir'])
        json.dump(params, open(args.tb_dir / run_name / 'params.json', 'w'))
        with open(args.dataset_directory / val_data.instances[0], 'rb') as file:
            G = pickle.load(file)
        H = val_data.get_scaled_features(G).to(args.device)
        x = H['node1'].x
        save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_final.pt')
        traced_model = torch.jit.trace(model, (list(H.edge_index_dict.values()), x))
        model_scripted = torch.jit.script(traced_model)
        model_scripted.save('modelv1.pt')
        epoch, train_loss, val_loss = load_checkpoint(model, optimizer, log_dir / 'checkpoint_best_val.pt')
        result = utils.atsp_results(model, args, val_data)
        formatted_result = {key: f'{(value/30):.4f}' for key, value in result.items()}  # Format values to 4 decimal places
        formatted_result['train_loss'] = f"{epoch_loss:.4f}"
        formatted_result['val_loss'] = f"{epoch_val_loss:.4f}"
        formatted_result['epoch'] = f'{epoch:.4f}'
        print('best epoch results')
        print(formatted_result)
if __name__ == '__main__':
    args = train_parse_args()
    run(args)