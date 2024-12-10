import os
import time
import signal
import logging
import warnings
import numpy as np
import sympy as sp
import matplotlib as mpl
import traceback
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from ND2.model import NDformer
from ND2.utils import init_logger, AutoGPU, get_fig, seed_all
from ND2.search import MCTS
from ND2.GDExpr import GDExpr
from ND2.dataset import Dataset
from ND2.search.reward_solver import RewardSolver

warnings.filterwarnings("ignore", category=RuntimeWarning)
mpl.rcParams['agg.path.chunksize'] = 100000

logger = logging.getLogger('ND2.search')
def handler(signum, frame): raise KeyboardInterrupt    
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

def main(args):
    seed_all(args.seed)
    init_logger(args.name, f'./log/search/{args.name}/info.log', root_name='ND2', quiet=args.quiet)
    setproctitle(f'{args.name}@ZihanYu')
    if args.device == 'auto':
        args.device = AutoGPU().choice_gpu(900, interval=15, force=True)

    logger.info(f'Args: {args}')
    logger.info(f'GDExpr.config: {GDExpr.config}')

    save_path = f'./result/search.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path): 
        with open(save_path, 'w') as f:
            f.write('\t'.join([
                'host', 'Name', 'Seed', 'Experiment', 'Idx', 'RMSE', 'MAPE', 'wMAPE', 'sMAPE', 
                'R2', 'ACC2', 'ACC3', 'ACC4', 'time', 'Final Time', 'equation', 'ground_truth'
            ]) + '\n')

    # %% Load Data
    # test_set = Dataset.load_csv(args.test_set, data_only=True, device=args.device, data_dir=args.data_dir)
    import json
    var_dict = json.load(open('./data/synthetic/kuramoto.json', 'r'))
    for k, v in var_dict.items():
        var_dict[k] = np.array(v)
    var_dict['A'] = var_dict['A'].astype(int)
    var_dict['G'] = var_dict['G'].astype(int)
    V = var_dict['A'].shape[0]
    E = var_dict['G'].shape[0]
    T = var_dict['x'].shape[0]

    # %% Init Model
    rewarder = RewardSolver(
        Xv={'omega': var_dict['omega'], 'x': var_dict['x']},
        Xe={},
        A=var_dict['A'],
        G=var_dict['G'],
        Y=var_dict['dx'],
        mask=None,
    )
    ndformer = NDformer(device=args.device)
    ndformer.load('./weights/Finetune-step2000.pth', weights_only=False)
    ndformer.eval()
    ndformer.set_data(
        Xv={'omega': var_dict['omega'].reshape(1, V).repeat(T, axis=0), 'x': var_dict['x']},
        Xe={},
        A=var_dict['A'],
        G=var_dict['G'],
        Y=var_dict['dx'],
        root_type='node',
        cache_data_emb=True
    )
    est = MCTS(
        rewarder=rewarder,
        ndformer=ndformer,
        vars_node=['x', 'omega'],
        vars_edge=[],
        # binary=['add', 'sub'],
        # unary=['sin', 'aggr', 'sour', 'targ'],
        # constant=[],
        log_per_episode=None,
        log_per_second=5,
        beam_size=10,
        use_random_simulate=False,
    )

    # %% Search
    # try:
    est.fit(['node']) 
    # except KeyboardInterrupt as e: 
    #     logger.info(f'Interrupted manually.')
    # except Exception:
    #     logger.error(traceback.format_exc())
    # finally:
    #     logger.note(est.best_result)
        # best_result = {**mcts.best_result, 'idx': test_idx, 'ground_truth': GDExpr.prefix2str(prefix)}
        # logger.note(f'MCTS Done. Expand {len(mcts.MC_Tree)} states in total. '
        #             f'Final best: ({best_result["reward"]:.2f}) '
        #             f'R2={best_result["R2"]:.4f}, '
        #             f'RMSE={best_result["RMSE"]:.3e}, '
        #             f'MAPE={best_result["MAPE"]:.2%}, '
        #             f'wMAPE={best_result["wMAPE"]:.2%}, '
        #             f'sMAPE={best_result["sMAPE"]:.2%}, '
        #             f'Acc2={best_result["ACC2"]:.2%}, '
        #             f'Acc3={best_result["ACC3"]:.2%}, '
        #             f'Acc4={best_result["ACC4"]:.2%}, '
        #             f'Complexity={best_result["complexity"]}, '
        #             f'Time={best_result["time"]:.2f}s, '
        #             f'[{best_result["equation"]}]')
        # with open(save_path, 'a') as f:
        #     f.write('\t'.join(map(str, [
        #         gethostname(), args.name, args.seed, test_set.datafiles[test_idx], best_result['idx'], 
        #         best_result['RMSE'], best_result['MAPE'], best_result['wMAPE'], best_result['sMAPE'], 
        #         best_result['R2'], best_result['ACC2'], best_result['ACC3'], best_result['ACC4'], 
        #         best_result['time'], time.time() - mcts.start_time, best_result['equation'], best_result['ground_truth']
        #     ])) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--args', type=str, default='./args/test.yaml', help='Path to the args file.')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Override seed in args.')
    parser.add_argument('-t', '--test_set', type=str, nargs='+', default=None, help='Override test_set in args.')
    parser.add_argument('-m', '--model_path', type=str, default=None, help='Override model_path in args.')
    parser.add_argument('-d', '--device', type=str, default=None, help='Override device in args.')
    parser.add_argument('-n', '--name', type=str, default=None, help='Override name in args.')
    parser.add_argument('-i', '--idx', type=int, nargs='+', default=[], help='只算 idx 中的实验')
    parser.add_argument('-b', '--beam-size', type=int, default=None, help='Override beam_size in args.')
    parser.add_argument('--split', action='store_true', help='see DEBUG_SPLIT_FLAG in mcts.py and model.py.')
    parser.add_argument('--noresolve', action='store_true', help='see DEBUG_NORESOLVE_FLAG in mcts.py.')
    parser.add_argument('--spurious_link_ratio', type=float, default=None, help='Override spurious_link_ratio in args.')
    parser.add_argument('--missing_link_ratio', type=float, default=None, help='Override missing_link_ratio in args.')
    parser.add_argument('--ObsSNR', type=float, default=None, help='Override noise in args.')
    parser.add_argument('--DynSNR', type=float, default=None)
    parser.add_argument('--diff_as_out', type=str, default=None, help='override diff_as_out in args.')
    parser.add_argument('--diff_mod_pi', action='store_true', help='override diff_mod_pi in args.')
    parser.add_argument('--quiet', action='store_true', help='Do not print log to console.')
    parser.add_argument('--use_json', action='store_true', help='Load data from xxx.json instead of xxx.pkl')
    parser.add_argument('--test_KURtest', action='store_true', help='Test KURtest dataset.')
    parser.add_argument('--tabula_rasa', action='store_true', help='Use tabular RASA.')
    args = parser.parse_args()

    # args = AttrDict.load_yaml(args.args)
    args.name = args.name or 'Search_' + time.strftime('%Y%m%d_%H%M%S')
    args.seed = args.seed or np.random.randint(1, 32768)
    # args.model_path = args.model_path or args.model_path
    # args.device = args.device or args.device
    # args.mcts.beam_size = args.beam_size or args.mcts.beam_size
    # args.mcts.ObsSNR = args.ObsSNR or args.mcts.ObsSNR
    # args.mcts.DynSNR = args.DynSNR or args.mcts.DynSNR
    # args.mcts.diff_as_out = args.diff_as_out or args.mcts.diff_as_out
    # args.mcts.diff_mod_pi = args.diff_mod_pi or args.mcts.diff_mod_pi
    # args.test_set = args.test_set or args.test_set
    # args.data_dir = os.path.dirname(args.test_set[0])
    # args.mcts.spurious_link_ratio = args.spurious_link_ratio or args.mcts.spurious_link_ratio
    # args.mcts.missing_link_ratio = args.missing_link_ratio or args.mcts.missing_link_ratio
    # args.mcts.tabula_rasa = args.tabula_rasa or args.mcts.tabula_rasa
    # if args.tabula_rasa: 
    #     args.mcts.model = None
    #     args.device = 'cpu'

    main(args)
