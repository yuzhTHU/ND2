import os
import json
import time
import signal
import logging
import warnings
import traceback
import numpy as np
import pandas as pd
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


class GeneRewardSolver(RewardSolver):
    first_time_log = True
    def solve(self, prefix, *args, **kwargs):
        if prefix.count('aggr') > 1: return 0.0, {}
        if prefix.count('aggr') == 1: 
            idx = prefix.index('aggr')
            prefix = prefix[:idx+1] + ['mul', '<Ce>'] + prefix[idx+1:]
            if self.first_time_log:
                logger.warning(f"""GeneRewardSolver modified the formula, changing aggr(...) to aggr(<Ce>*...), but this modification is transparent to MCTS. Please pay attention to the way you understand the final formula.""")
                self.first_time_log = False
        reward, prefix_with_coef = super().solve(prefix, *args, **kwargs)
        return reward, prefix_with_coef


def main(args):
    init_logger(args.name, f'./log/gene/{args.name}/info.log', root_name='ND2', info_level=args.info_level)
    setproctitle(f'{args.name}@ZihanYu')
    if args.seed is None: args.seed = np.random.randint(0, 32768)
    seed_all(args.seed)
    if args.device == 'auto': args.device = AutoGPU().choice_gpu(900, interval=15, force=True)
    logger.info(f'Args: {args}')

    # %% Load Data
    data = pd.read_csv('./data/gene/data.csv')
    data = np.array(data)[:, 1:]
    B = 1
    T = data.shape[0] - 1 # 17 - 1 = 16
    V = data.shape[1] # 7
    E = V * (V - 1) # 42
    A = 1 - np.eye(V, dtype=int) # (V, V)
    G = np.stack(np.nonzero(A), axis=-1) # (E, 2)
    x = data[:-1, :] # (T, V)
    y = np.diff(data, axis=0) # (T, V)
    data = dict(A=A, G=G, y=y, x=x)

    # %% Init Model
    rewarder = GeneRewardSolver(
        Xv={'x': data['x']},
        Xe={},
        A=data['A'],
        G=data['G'],
        Y=data['y'],
        mask=None,
    )
    ndformer = NDformer(device=args.device)
    ndformer.load(args.model_path, weights_only=False)
    ndformer.eval()
    ndformer.set_data(
        Xv={'x': data['x']},
        Xe={},
        A=data['A'],
        G=data['G'],
        Y=data['y'],
        root_type='node',
        cache_data_emb=True
    )
    est = MCTS(
        rewarder=rewarder,
        ndformer=ndformer,
        vars_node=['x'],
        vars_edge=[],
        # binary=['add', 'sub'],
        # unary=['sin', 'aggr', 'sour', 'targ'],
        # constant=[],
        log_per_episode=None,
        log_per_second=10,
        beam_size=10,
        use_random_simulate=False,
        max_token_num=30,
        max_coeff_num=5,
    )

    # Evaluate Existing Model as Baseline
    # f = '<Cv> - <Cv> * x + aggr(regular(sour(<Cv>*x), 2))'
    f = '<C> - <C> * x + aggr(regular(sour(<C>*x), 2))'
    prefix = GDExpr.sympy2prefix(GDExpr.parse_expr(f), 'node', reindex=False)
    reward, prefix_with_coef = rewarder.solve(prefix, sample=False, max_iter=100)
    metrics = rewarder.evaluate(prefix_with_coef, {})
    log = { 'reward': reward, 'Baseline': f, **metrics, }
    logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))

    # %% Search
    try:
        logger.note('Start searching... Press ^C (Ctrl+C) to stop when you fell satisfied.')
        est.fit(
            # ['node'], 
            ['add', 'sub', '<C>', 'mul', '<C>', 'x', 'node'],
            episode_limit=100_000_000, 
            time_limit=args.time_limit,
            early_stop=None,
        )
    except KeyboardInterrupt:
        logger.info(f'Interrupted manually.')
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        log = {
            'Discovered': GDExpr.prefix2str(est.best_model),
            **est.best_metric,
        }
        logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))
        pareto = est.Pareto()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=f'GeneSearch_{time.strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('-d', '--device', type=str, default='auto')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--model_path', type=str, default='./weights/checkpoint.pth')
    parser.add_argument('--time_limit', type=int, default=86400)
    parser.add_argument('--info_level', choices=['debug', 'info', 'note', 'warning', 'error', 'critical'], default='note')
    args, unknown = parser.parse_known_args()
    if unknown: warnings.warn(f'Unknown args: {unknown}')
    main(args)
