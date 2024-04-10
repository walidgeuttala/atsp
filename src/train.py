import tqdm.auto as tqdm
import argparse
import logging
import datetime
import uuid
import json
import pathlib
import os 

import torch 
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import get_model
import dataset

def train(model, data_loader, criterion, optimizer, args):
    model.train()

    epoch_loss = 0
    for batch_i, batch in enumerate(data_loader):
        batch = batch.to(args.device)
        x = batch.x
        y = batch.y
        optimizer.zero_grad()
        y_pred = model(x, batch.edge_index)
        loss = criterion(y_pred, y.type_as(y_pred))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (batch_i + 1)
    return epoch_loss

def test(model, data_loader, criterion, args):
    with torch.no_grad():
        model.eval()

        epoch_loss = 0
        for batch_i, batch in enumerate(data_loader):
            batch = batch.to(args.device)
            x = batch.x
            y = batch.y

            y_pred = model(x, batch.edge_index)
            loss = criterion(y_pred, y.type_as(y_pred))

            epoch_loss += loss.item()

        epoch_loss /= (batch_i + 1)
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
    parser.add_argument("--dataset_directory", type=pathlib.Path, help="Directory to save datasets", default="../../tsp_n5900")
    parser.add_argument("--tb_dir", type=pathlib.Path, help="Directory to save checkpoints", default="../../checkpoint")

    ### Model Args
    parser.add_argument("--model", type=str, help="Model type", default="gnn")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
    parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=1)
    parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.0)
    parser.add_argument("--alpha", type=float, help="Direction convex combination params", default=0.5)
    parser.add_argument("--learn_alpha", action="store_true")
    parser.add_argument("--conv_type", type=str, help="DirGNN Model", default="sage")
    parser.add_argument("--gat_model", type=bool, help="GAT with skip connetion", default=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--jk", type=str, choices=["max", "cat", False], default="max")
    parser.add_argument('--num_features', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--target', type=str, default='regret')

    ### Training Args
    parser.add_argument("--lr_init", type=float, help="Learning Rate", default=0.01)
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.001)
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument("--patience", type=int, help="Patience for early stopping", default=10)
    parser.add_argument("--num_runs", type=int, help="Max number of runs", default=1)
    parser.add_argument('--checkpoint_freq', type=int, default=20, help='Checkpoint frequency')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Early stopping min delta')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')

    ### System Args
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--num_workers", type=int, help="Num of workers for the dataloader", default=16)

    args = parser.parse_args()

    return args

def run(args):
    torch.manual_seed(0)
    
    for _ in range(args.num_runs):
         
        train_data = dataset.TSPDataset(f"{args.dataset_directory}/train.txt")
        val_data = dataset.TSPDataset(f"{args.dataset_directory}/val.txt")

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

        model = get_model(args).to(args.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.weight_decay)


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

        pbar = tqdm.trange(args.n_epochs)
        for epoch in pbar:
            epoch_loss = train(model, train_loader, criterion, optimizer, args)
            writer.add_scalar("Loss/train", epoch_loss, epoch)

            epoch_val_loss = test(model, val_loader, criterion, args)
            writer.add_scalar("Loss/validation", epoch_val_loss, epoch)

            pbar.set_postfix({
                'Train Loss': '{:.4f}'.format(epoch_loss),
                'Validation Loss': '{:.4f}'.format(epoch_val_loss),
            })
            
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

        save(model, optimizer, epoch, epoch_loss, epoch_val_loss, log_dir / 'checkpoint_final.pt')


if __name__ == '__main__':
    args = train_parse_args()
    run(args)