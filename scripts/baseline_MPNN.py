import os
import json
import time
import torch
import signal
import logging
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GCNConv, MessagePassing
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from setproctitle import setproctitle
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from ND2.utils import init_logger, seed_all, AutoGPU
from ND2.search.reward_solver import RewardSolver
from ND2.GDExpr import GDExpr
from typing import Dict, List

warnings.filterwarnings("ignore", category=RuntimeWarning)
def handler(signum, frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
logger = logging.getLogger('ND2.search')


class MLP(nn.Module):
    def __init__(self, *dims, dropout=0.0, act=nn.ReLU, act_out=nn.Identity):
        super(MLP, self).__init__()
        seq = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            seq.append(nn.Linear(d_in, d_out))
            if i < len(dims) - 2:
                # seq.append(nn.LayerNorm())
                seq.append(act())
                seq.append(nn.Dropout(dropout))
        seq.append(act_out())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class GNN(MessagePassing):
    def __init__(self, edge_index, hidden_dim, dropout=0.0, act=nn.ReLU):
        super(GNN, self).__init__(aggr='add')
        self.edge_index = edge_index
        self.node_num = edge_index.max().item() + 1
        self.edge_num = edge_index.shape[1]
        # heterogeneous node parameters
        self.theta = nn.Parameter(torch.rand(self.node_num, hidden_dim[0]), requires_grad=True)
        # heterogeneous edge parameters
        self.weight = nn.Parameter(torch.rand(self.edge_num, hidden_dim[0]), requires_grad=True)
        self.F = MLP(2*hidden_dim[0], *hidden_dim, act=act, dropout=dropout)
        self.G = MLP(3*hidden_dim[0], *hidden_dim, act=act, dropout=dropout)

    def forward(self, x):
        x = self.F(torch.cat([x, self.theta.expand_as(x)], dim=-1)) + self.propagate(self.edge_index, x=x, w=self.weight)
        return x
    
    def message(self, x_i, x_j, w):
        return self.G(torch.cat([x_i, x_j, w.expand_as(x_i)], dim=-1))

# Model
class MPNN(nn.Module):
    def __init__(self, edge_index, hidden_dim, dropout=0.0):
        super(MPNN, self).__init__()
        self.embed = MLP(1, hidden_dim[0])
        self.gnn = GNN(edge_index, hidden_dim, dropout=dropout, act=nn.ELU)
        self.out = MLP(hidden_dim[-1], 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x)
        x = self.gnn(x)
        x = self.out(x)
        x = x.squeeze(-1)
        return x


def main(args):
    start_time = time.time()
    # %% Load Data & Init Model
    if args.data == 'ecological':
        data_path = './data/bacteria/'
        data_list = [
            *list(np.array(eval(open(f'{data_path}/low/numes_6.json').read()))[(2,), 0, :, :]),
            # *list(np.array(eval(open(f'{data_path}/low/numes_6.json').read()))[(2,3), 0, :, :])
        ] # List of np.ndarray(T+1, V), T=10, V=[3,6,12,24]

        data = np.concatenate(data_list, axis=-1) # (T, V_sum)
        x = data[:-1, :] # (T, V_sum)
        dx = np.diff(data, axis=0) # (T, V_sum)
        T, V = x.shape
        A = np.zeros((V, V), dtype=int)
        tmp = np.cumsum([0, *[d.shape[-1] for d in data_list]])
        for i, j in zip(tmp[:-1], tmp[1:]): A[i:j, i:j] = 1 - np.eye(j-i)
        G = np.stack(np.nonzero(A), axis=-1) # (E, 2)
        E = G.shape[0]
        data = dict(A=A, G=G, dx=dx, x=x)
        assert E == sum(d.shape[-1] * (d.shape[-1] - 1) for d in data_list)
    else:
        df = pd.read_csv('./data/gene/data.csv').set_index('Time', drop=True)
        data = df.values

        T = data.shape[0] - 1 # 17 - 1 = 16
        V = data.shape[1] # 7
        E = V * (V - 1) # 42
        A = 1 - np.eye(V) # (V, V)
        G = np.stack(np.nonzero(A), axis=-1) # (E, 2)

        x = np.array(data[:-1, :]) # (T, V)
        dx = np.diff(data, axis=0) # (T, V)
        data = dict(V=V, E=E, A=A, G=G, dx=dx, x=x)

    # init Dataset
    dataset = Data(x=torch.tensor(data['x'], dtype=torch.float32),
                   dx=torch.tensor(data['dx'], dtype=torch.float32),
                   edge_index=torch.tensor(G.T, dtype=torch.long))
    dataset = dataset.to(args.device)
    index_dataset = TensorDataset(torch.arange(T))
    loader = DataLoader(index_dataset, batch_size=args.batch_size, shuffle=True)

    # init Model
    model = MPNN(dataset.edge_index, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    criterion = torch.nn.MSELoss()
    
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {count_trainable_parameters(model)}')

    # %% Train
    best_metric, patience = {'RMSE': float('inf')}, args.patience
    for epoch in range(args.epoch_num):
        # Train
        model.train()
        loss_list = []
        for index in loader:
            index = index[0].to(args.device)
            x = dataset.x[index, :]
            dx = dataset.dx[index, :]
            out = model(x)
            loss = criterion(out, dx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_list.append(loss.item())
        loss = np.mean(loss_list)
        logger.info(f'Epoch {epoch}, Loss: {loss}')

        # Evaluate
        model.eval()
        pred = model(dataset.x).double().detach().cpu().numpy()
        true = dataset.dx.double().detach().cpu().numpy()
        ss_RMSE = np.sqrt(np.mean((pred - true) ** 2))
        ss_sMAPE = 2 * np.mean(np.abs(pred - true) / (np.abs(pred) + np.abs(true)))

        with torch.no_grad():
            true = np.concatenate([data['x'], data['x'][(-1,), :] + data['dx'][(-1,), :]], axis=0)
            pred = [data['x'][[0]]]
            N = 10
            for t in range(T):
                for n in range(N):
                    x = torch.from_numpy(pred[-1]).float().to(args.device)
                    dx = model(x).double().detach().cpu().numpy()
                    x_ = pred[-1] + dx / N
                    x_ = x_.clip(0, 1)
                    pred.append(x_)
        pred = np.concatenate(pred, axis=0)
        RMSE = np.sqrt(np.mean((pred[::N] - true) ** 2))
        sMAPE = np.mean(np.abs(pred[::N] - true) / (np.abs(pred[::N]) + np.abs(true)))
        logger.info(f'Epoch {epoch}, ss_RMSE: {ss_RMSE:.4f}, ss_sMAPE: {ss_sMAPE:.2%}, RMSE: {RMSE:.4f}, sMAPE: {sMAPE:.2%}')

        ## Plot
        if args.plot_every_epoch and epoch % args.plot_every_epoch == 0:
            fig = plt.figure()
            plt.plot(np.arange(T+1), true, 'o:', label='True', color='gray')
            for i in range(pred.shape[-1]):
                plt.plot(np.arange(T*N+1)/N, pred[:, i], label='Pred', color=f'C{i}')
                plt.plot(np.arange(T+1), pred[::N, i], 'o', color=f'C{i}')
            plt.title(f'Epoch {epoch}, RMSE: {RMSE:.4f}, sMAPE: {sMAPE:.2%}')
            plt.savefig(f'./log/baseline/{args.name}/epoch_{epoch}.png')
            plt.close(fig)

        # Early stopping
        if RMSE < best_metric['RMSE']:
            best_metric['loss'] = loss
            best_metric['ss_RMSE'] = ss_RMSE
            best_metric['ss_sMAPE'] = ss_sMAPE
            best_metric['RMSE'] = RMSE
            best_metric['sMAPE'] = sMAPE
            best_metric['epoch'] = epoch
            patience = args.patience
        elif patience > 1:
            patience -= 1
        else:
            logger.note(f'Early stopping at epoch {epoch}')
            break

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    from socket import gethostname
    with open(args.save_path, 'a') as f:
        json.dump(dict(
            host=gethostname(),
            lr=args.lr,
            batch_size=args.batch_size,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
            name=args.name,
            time=time.time() - start_time,
            **best_metric,
            final_loss=loss,
            final_ssRMSE=ss_RMSE,
            final_sssMAPE=ss_sMAPE,
            final_RMSE=RMSE,
            final_sMAPE=sMAPE,
        ), f)
        f.write('\n')

    logger.note(f'Best RMSE: {best_metric["RMSE"]:.4f}, sMAPE: {best_metric["sMAPE"]:.2%}, loss: {best_metric["loss"]:.4f} ss_RMSE: {best_metric["ss_RMSE"]}, ss_sMAPE: {best_metric["ss_sMAPE"]}, at epoch {best_metric["epoch"]}')
    return best_metric


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=f'MPNN_{time.strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('-d', '--device', type=str, default='auto')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--info_level', choices=['debug', 'info', 'note', 'warning', 'error', 'critical'], default='info')
    parser.add_argument('--epoch_num', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--plot_every_epoch', type=int, default=None)
    parser.add_argument('--data', type=str, default='ecological', choices=['ecological', 'gene'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--save_path', type=str, default='./result/baseline_MPNN.csv')
    args, unknown = parser.parse_known_args()
    if unknown: 
        warnings.warn(f'Unknown args: {unknown}')
    init_logger(args.name, f'./log/baseline/{args.name}/info.log', root_name='ND2', info_level=args.info_level)
    setproctitle(f'{args.name}@ZihanYu')
    if args.seed is None: 
        args.seed = np.random.randint(0, 32768)
    seed_all(args.seed)
    if args.device == 'auto': 
        args.device = AutoGPU().choice_gpu(400, interval=15, force=False)
    logger.info(f'Args: {args}')

    main(args)