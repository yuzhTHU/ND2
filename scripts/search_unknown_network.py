import os
import json
import time
import signal
import logging
import warnings
import numpy as np
import traceback
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from ND2.model import NDformer
from ND2.utils import init_logger, AutoGPU, seed_all
from ND2.search import MCTS
from ND2.GDExpr import GDExpr
from ND2.search.reward_solver import RewardSolver

warnings.filterwarnings("ignore", category=RuntimeWarning)
def handler(signum, frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
logger = logging.getLogger('ND2.search')


from copy import deepcopy
from scipy.optimize import minimize
class RewardSolver_with_LinearRegression(RewardSolver):
    def solve(self, prefix, sample=False, x0=None, method='L-BFGS-B', max_iter=1000, **kwargs):
        if kwargs: logger.warning(f'Unused arguments: {kwargs} in RewardSolver.solve')

        if prefix.count('aggr') == 0:
            return super().solve(prefix, sample=sample, x0=x0, method=method, max_iter=max_iter, **kwargs)
        elif prefix.count('aggr') > 1:
            return 0, prefix
        else:
            if prefix[0] != 'add': return 0, prefix
            elif prefix[1] == 'aggr': return 0, prefix
            else: pass

        aggr_index = prefix.index('aggr')
        F_prefix = prefix[1:aggr_index] # node
        G_prefix = prefix[aggr_index + 1:] # edge

        V, E = self.A.shape[0], self.G.shape[0]
        F_num_C, F_num_Cv, F_num_Ce = F_prefix.count('<C>'), F_prefix.count('<Cv>'), F_prefix.count('<Ce>')
        G_num_C, G_num_Cv, G_num_Ce = G_prefix.count('<C>'), G_prefix.count('<Cv>'), G_prefix.count('<Ce>')
        num_C, num_Cv, num_Ce = F_num_C + G_num_C, F_num_Cv + G_num_Cv, F_num_Ce + G_num_Ce

        sample = False
        N = int(np.ceil(self.sample_num / self.Y.shape[1]))
        T = self.Y.shape[0]
        if sample and (N < T):
            sample_idx = np.random.choice(T, N, replace=False)
            var_dict = {'A': self.A, 'G': self.G, 'out': self.Y[sample_idx]} | \
                       {k: (v[sample_idx] if v.ndim > 1 else v) for k, v in self.Xv.items()} | \
                       {k: (v[sample_idx] if v.ndim > 1 else v) for k, v in self.Xe.items()}
            Y = var_dict['out']
        else:
            var_dict = self.var_dict
            Y = self.Y

        def params2coefdict(params):
            coef_dict = {
                '<C>': params[:num_C],
                '<Cv>': params[num_C:num_C + num_Cv * V].reshape(num_Cv, V),
                '<Ce>': params[num_C + num_Cv * V:].reshape(num_Ce, E)
            }
            F_coef_dict = {
                '<C>': coef_dict['<C>'][:F_num_C],
                '<Cv>': coef_dict['<Cv>'][:F_num_Cv, :],
                '<Ce>': coef_dict['<Ce>'][:F_num_Ce, :],
            }
            G_coef_dict = {
                '<C>': coef_dict['<C>'][F_num_C:],
                '<Cv>': coef_dict['<Cv>'][F_num_Cv:, :],
                '<Ce>': coef_dict['<Ce>'][F_num_Ce:, :],
            }
            return F_coef_dict, G_coef_dict

        def loss(params, return_w=False):
            global w
            F_coef_dict, G_coef_dict = params2coefdict(params)
            F_prefix_with_coef = deepcopy(F_prefix)
            G_prefix_with_coef = deepcopy(G_prefix)
            tmp = {k: list(v) for k, v in F_coef_dict.items()}
            F_prefix_with_coef = [tmp[token].pop(0) if token in tmp else token for token in F_prefix_with_coef]
            tmp = {k: list(v) for k, v in G_coef_dict.items()}
            G_prefix_with_coef = [tmp[token].pop(0) if token in tmp else token for token in G_prefix_with_coef]

            F_pred = GDExpr.eval(F_prefix_with_coef, var_dict, [], strict=False) # (T, V)
            G_pred = GDExpr.eval(G_prefix_with_coef, var_dict, [], strict=False) # (T, E)

            T = Y.shape[0]
            if np.size(F_pred) == 1:
                F_pred = np.full((T, V), F_pred)
            elif F_pred.ndim == 1 and F_pred.shape[0] == T:
                F_pred = F_pred[:, np.newaxis].repeat(V, axis=1)
            if np.size(G_pred) == 1:
                G_pred = np.full((T, E), G_pred)
            elif G_pred.ndim == 1 and G_pred.shape[0] == T:
                G_pred = G_pred[:, np.newaxis].repeat(E, axis=1)
            elif G_pred.ndim == 1 and G_pred.shape[0] == E:
                G_pred = G_pred[np.newaxis, :].repeat(T, axis=0)
            elif G_pred.ndim == 2 and G_pred.shape[0] == 1 and T > 1:
                G_pred = G_pred.repeat(T, axis=0)

            X = G_pred
            y = Y - F_pred
            index = self.G[:, 1]
            w = np.zeros((E,))
            for i in range(V):
                e_idx = np.where(index == i)[0]
                if not np.isfinite(X[:, e_idx]).all(): continue
                if not np.isfinite(y[:, i]).all(): continue
                w[e_idx] = np.linalg.lstsq(X[:, e_idx], y[:, i], rcond=None)[0]
                # print(X[:, e_idx].shape, y[:, i].shape, w[e_idx].shape)
            pred = GDExpr.eval(['add', 'F_pred', 'aggr', 'mul', 'w', 'G_pred'], 
                               {'F_pred': F_pred, 'G_pred': G_pred, 'w': w, 
                                'A': var_dict['A'], 'G': var_dict['G']}, [], strict=False)

            true = Y
            if return_w: return np.mean((pred - true) ** 2), w
            return np.mean((pred - true) ** 2)

        if num_C + num_Cv + num_Ce == 0:
            MSE, w = loss(np.empty(0), return_w=True)
            prefix_with_coef = ['add', *F_prefix, 'aggr', 'mul', w, *G_prefix]
        else:
            x0 = x0 or {'<C>': np.random.randn(num_C),
                        '<Cv>': np.random.randn(num_Cv, V),
                        '<Ce>': np.random.randn(num_Ce, E)}
            x0 = np.concatenate([x0['<C>'], x0['<Cv>'].reshape(-1), x0['<Ce>'].reshape(-1)])
            res = minimize(loss, x0, method=method, options={'maxiter': max_iter})
            MSE, w = loss(res.x, return_w=True)
            F_coef_dict, G_coef_dict = params2coefdict(res.x)
            tmp = {k: list(v) for k, v in F_coef_dict.items()}
            F_prefix_with_coef = deepcopy(F_prefix)
            F_prefix_with_coef = [tmp[token].pop(0) if token in tmp else token for token in F_prefix_with_coef]
            tmp = {k: list(v) for k, v in G_coef_dict.items()}
            G_prefix_with_coef = deepcopy(G_prefix)
            G_prefix_with_coef = [tmp[token].pop(0) if token in tmp else token for token in G_prefix_with_coef]
            prefix_with_coef = ['add', *F_prefix_with_coef, 'aggr', 'mul', w, *G_prefix_with_coef]

        if not np.isfinite(MSE): MSE = np.inf
        n = Y.size
        sigma = 1e-4
        threshold = 1e-2
        k = np.sum(w > threshold)
        MSE += sigma ** 2 * np.log(n) / n * k
        r_MSE = MSE / self.var_out.clip(1e-7)
        if r_MSE < 1e-3: reward = 1 - r_MSE
        else: reward = self.complexity_base ** len(prefix) / (1 + r_MSE)
        return reward, prefix_with_coef


def main(args):
    # %% Load Data & Init Model
    data = json.load(open(args.data, 'r'))
    for k, v in data.items():
        data[k] = np.array(v)
    data['A'] = data['A'].astype(int)
    data['G'] = data['G'].astype(int)
    
    # Make network A & G unknown
    data['A'] = np.ones_like(data['A']) - np.eye(data['A'].shape[0])
    data['G'] = np.stack(np.nonzero(data['A']), axis=-1)

    # init Rewarder
    rewarder = RewardSolver_with_LinearRegression(
        Xv={var: data[var] for var in args.vars},
        Xe={},
        A=data['A'],
        G=data['G'],
        Y=data[args.target_var],
    )

    # init NDformer
    ndformer = NDformer(device=args.device)
    ndformer.load(args.ndformer_path, weights_only=False)
    ndformer.eval()
    ndformer.set_data(
        Xv={var: data[var] for var in args.vars},
        Xe={},
        A=data['A'],
        G=data['G'],
        Y=data[args.target_var],
        root_type='node',
        cache_data_emb=True,
        # encode_with_cpu=False,
    )

    # init Monte-Carlo Tree Search algorithm
    est = MCTS(
        rewarder=rewarder,
        ndformer=ndformer,
        vars_node=args.vars,
        vars_edge=[],
        # binary=['add', 'sub'],
        # unary=['sin', 'aggr', 'sour', 'targ'],
        # constant=[],
        log_per_episode=10,
        log_per_second=None,
        beam_size=10,
        use_random_simulate=False,
        max_coeff_num=0,
    )

    # %% Search
    try:
        est.fit(['add', 'node', 'aggr', 'edge']) # Search for formula with form of "F + Aggr(G)"
    except KeyboardInterrupt as e: 
        logger.info(f'Interrupted manually.')
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        logger.note(f'Search finished. Discovered model: {GDExpr.prefix2str(est.best_model)}')
        logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in est.best_metric.items()))

        for idx, token in enumerate(est.best_model):
            if isinstance(token, np.ndarray):
                est.best_model[idx] = token.tolist()

        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, 'a') as f:
            json.dump(dict(
                host=gethostname(),
                name=args.name,
                seed=args.seed,
                result=est.best_model,
                **est.best_metric,
            ), f)
            f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=f'SearchUnknownNetwork_{time.strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('-d', '--device', type=str, default='auto')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--data', type=str, default='./data/unknown_network/kuramoto_BA10.json')
    parser.add_argument('--info_level', choices=['debug', 'info', 'note', 'warning', 'error', 'critical'], default='info')
    parser.add_argument('--ndformer_path', type=str, default='./weights/checkpoint.pth')
    parser.add_argument('--vars', type=str, nargs='*', default=['x', 'omega'])
    parser.add_argument('--target_var', type=str, default='dx')
    parser.add_argument('--save_path', type=str, default='./result/search.csv')
    args, unknown = parser.parse_known_args()
    if unknown: 
        warnings.warn(f'Unknown args: {unknown}')
    init_logger(args.name, f'./log/search/{args.name}/info.log', root_name='ND2', info_level=args.info_level)
    setproctitle(f'{args.name}@ZihanYu')
    if args.seed is None: 
        args.seed = np.random.randint(0, 32768)
    seed_all(args.seed)
    if args.device == 'auto': 
        args.device = AutoGPU().choice_gpu(3500, interval=15, force=True)
    logger.info(f'Args: {args}')

    main(args)
