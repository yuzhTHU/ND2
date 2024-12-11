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


class EpidemicRewardSolver(RewardSolver):
    def solve(self, prefix, *args, **kwargs):
        if prefix.count('aggr') != 1: return 0.0, {}
        reward, coef_dict = super().solve(prefix, *args, **kwargs)
        return reward, coef_dict


def main(args):
    init_logger(args.name, f'./log/epidemic/{args.name}/info.log', root_name='ND2', info_level=args.info_level)
    setproctitle(f'{args.name}@ZihanYu')
    if args.seed is None: args.seed = np.random.randint(0, 32768)
    seed_all(args.seed)
    if args.device == 'auto': args.device = AutoGPU().choice_gpu(900, interval=15, force=True)
    logger.info(f'Args: {args}')

    # %% Load Data
    data = json.load(open(f'./data/epidemic/COVID19in{args.data}.json'))
    case = np.array(data['case'], dtype=np.float32)
    # population = np.array(data['population'], dtype=np.float32)
    flow = np.array(data['flow'], dtype=np.float32)[np.newaxis, :].repeat(case.shape[0]-1, axis=0)
    A = np.array(data['A'])
    G = np.array(data['G'])
    y = case[1:].copy()
    X7 = case[:-1].copy()
    X14 = case[:-1].copy(); X14[1:] += case[:-2]
    # P = population.copy()
    F = flow.copy()
    data = dict(
        A=A,
        G=G,
        y=y/y.max(),
        x=X7/X7.max(),
        x2=X14/X14.max(),
        # v3=P,
        M=F/F.max(),
    )

    # %% Init Model
    rewarder = EpidemicRewardSolver(
        Xv={'x': data['x'], 'x2':data['x2']},
        Xe={'M': data['M']},
        A=data['A'],
        G=data['G'],
        Y=data['y'],
        mask=None,
    )
    ndformer = NDformer(device=args.device)
    ndformer.load(args.model_path, weights_only=False)
    ndformer.eval()
    ndformer.set_data(
        Xv={'x': data['x'], 'x2':data['x2']},
        Xe={'M': data['M']},
        A=data['A'],
        G=data['G'],
        Y=data['y'],
        root_type='node',
        cache_data_emb=True
    )
    est = MCTS(
        rewarder=rewarder,
        ndformer=ndformer,
        vars_node=['x', 'x2'],
        vars_edge=['M'],
        binary=['add', 'sub', 'mul', 'div', 'regular'],
        unary=['neg', 'abs', 'inv', 'exp', 'logabs', 'sqrtabs', 'pow2', 'pow3', 'tanh', 'sigmoid', 'aggr', 'sour', 'targ'],
        # constant=[],
        log_per_episode=None,
        log_per_second=10,
        beam_size=10,
        use_random_simulate=False,
        max_token_num=30,
        max_coeff_num=5,
    )

    # Evaluate Existing Model as Baseline
    # f = 'x'
    # prefix = GDExpr.sympy2prefix(GDExpr.parse_expr(f), 'node', reindex=False)
    # metrics = rewarder.evaluate(prefix, {})
    # log = { 'Baseline': f, **metrics, }
    # logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))

    # %% Search
    try:
        logger.note('Start searching... Press ^C (Ctrl+C) to stop when you fell satisfied.')
        est.fit(
            # ['node'], 
            ['node'],
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
    parser.add_argument('-n', '--name', type=str, default=f'EpidemicSearch_{time.strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('-d', '--device', type=str, default='auto')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--data', choices=['NYC', 'CHI', 'NYS', 'ILS', 'US', 'CN', 'World'], default='World')
    parser.add_argument('--model_path', type=str, default='./weights/checkpoint.pth')
    parser.add_argument('--time_limit', type=int, default=86400)
    parser.add_argument('--info_level', choices=['debug', 'info', 'note', 'warning', 'error', 'critical'], default='note')
    args, unknown = parser.parse_known_args()
    if unknown: warnings.warn(f'Unknown args: {unknown}')
    main(args)
