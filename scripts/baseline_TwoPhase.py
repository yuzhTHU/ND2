import os
import json
import time
import random
import socket
import signal
import logging
import warnings
from setproctitle import setproctitle
import numpy as np
from ND2.dataset import Dataset
from ND2.GDExpr import GDExpr
from ND2.utils import AttrDict, init_logger, seed_all
from baseline.NCS22.code.utils.ElementaryFunctions_Matrix import ElementaryFunctions_Matrix
from baseline.NCS22.code.utils.TwoPhaseInference import *
from argparse import ArgumentParser

warnings.filterwarnings("ignore", category=RuntimeWarning)
def handler(signum, frame): raise KeyboardInterrupt
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
logger = logging.getLogger('ND2.baseline_NCS22')


def run_two_phase(Timeseries, targets, Nnodes, A, args):
    # generate function library
    func_lib = ElementaryFunctions_Matrix(Timeseries, args.vars, Nnodes, A,
                                          polynomial_order=args.polynomial_order,
                                          activation_alpha=args.activation_alpha,
                                          activation_beta=args.activation_beta,
                                          activation_gamma=args.activation_gamma,
                                          coupled_polynomial_order=args.coupled_polynomial_order,
                                          coupled_activation_alpha=args.coupled_activation_alpha,
                                          coupled_activation_beta=args.coupled_activation_beta,
                                          coupled_activation_gamma=args.coupled_activation_gamma,
                                          PolynomialIndex='Polynomial' in args.library, 
                                          TrigonometricIndex='Trigonometric' in args.library,
                                          ExponentialIndex='Exponential' in args.library,
                                          FractionalIndex='Fractional' in args.library,
                                          ActivationIndex='Activation' in args.library,
                                          RescalingIndex='Rescaling' in args.library,
                                          CoupledPolynomialIndex='CoupledPolynomial' in args.library,
                                          CoupledTrigonometricIndex='CoupledTrigonometric' in args.library,
                                          CoupledExponentialIndex='CoupledExponential' in args.library,
                                          CoupledFractionalIndex='CoupledFractional' in args.library,
                                          CoupledActivationIndex='CoupledActivation' in args.library,
                                          CoupledRescalingIndex='CoupledRescaling' in args.library)
    logger.info(f'Function library shape: {func_lib.shape}')
    logger.debug('Function library: {' + ', '.join(map(str, list(func_lib.columns))) + '}')

    # drop nan in the function library
    Xy = pd.concat([func_lib, targets],axis=1)
    Xy = Xy.mask(Xy.abs() > 1e100).dropna()
    Xy = Xy.dropna()
    func_lib = Xy.iloc[:, :-1]
    targets = Xy.iloc[:, -1:]
    logger.info(f'Function library shape after dropping NaN: {func_lib.shape} (targets: {targets.shape})')

    # inference function combination
    Keep = 10
    Batchsize = min(10, Nnodes)
    SampleTimes = 20
    Lambda = pd.DataFrame(0.1*np.ones((1, 3)))
    InferredResults, _, _, intercept = TwoPhaseInference(func_lib, targets, Nnodes, 0, 1, Keep, SampleTimes, Batchsize, Lambda, 0.5, 0.7, verbose=args.verbose)
    logger.info(f'InferredResults shape: {InferredResults.shape}')
    logger.debug(f'InferredResults (first 5 columns):\n{InferredResults.T.head(5).T}')

    return func_lib, InferredResults, intercept

def main(args):
    start_time = time.time()

    # Load data
    data = json.load(open(args.data, 'r'))
    for k, v in data.items():
        data[k] = np.array(v).astype(float)
    data['A'] = data['A'].astype(int)
    data['G'] = data['G'].astype(int) # G = np.stack(np.nonzero(A), axis=-1) can be obtained from A

    # prepare input data
    Timeseries = np.stack([data[var] + 0 * data[args.target_var] for var in args.vars], axis=-1) # (Nsamples, Nnodes, Dim)
    Timelength, Nnodes, Dim = Timeseries.shape
    Timeseries = Timeseries.reshape(Timelength, Nnodes*Dim)
    targets = pd.DataFrame({args.target_var: data[args.target_var].T.flatten()})
    A = data['A'].T # since it assumes that A[j,i]=A_ij

    # add noise
    if args.obs_noise_SNR is not None:
        signal_scale = np.std(targets.values)
        noise_scale = np.sqrt(signal_scale ** 2 / (10 ** (args.obs_noise_SNR / 10)))
        targets += noise_scale * np.random.normal(0, 1, targets.shape)
        logger.note(f'Added observation noise with SNR={args.obs_noise_SNR} dB, noise scale: {noise_scale:.4f}, signal scale: {signal_scale:.4f}')

    # add missing links
    if args.missing_link_ratio is not None:
        E = data['G'].shape[0]
        remove = np.random.choice(E, int(E * args.missing_link_ratio), replace=False)
        A[data['G'][remove, 0], data['G'][remove, 1]] = 0
        logger.note(f'Removed {np.sum(A)/E:.2%} links, total links: {E}->{np.sum(A)}')

    # add spurious links
    if args.spurious_link_ratio is not None:
        E = data['G'].shape[0]
        anti_A = 1 - data['A']
        anti_A -= np.diag(np.diag(anti_A))  # remove self-loops
        anti_G = np.stack(np.nonzero(anti_A), axis=-1)
        anti_E = anti_G.shape[0]
        add = np.random.choice(anti_E, int(E * args.spurious_link_ratio), replace=False)
        A[anti_G[add, 0], anti_G[add, 1]] = 1
        logger.note(f'Added {np.sum(A)/E-1:.2%} spurious links, total links: {E}->{np.sum(A)}')

    # infer function combinations
    func_lib, InferredResults, intercept = run_two_phase(Timeseries, targets, Nnodes, A, args)

    # Post-process to generate the equation
    equation = '+'.join(["{}*{}".format(coef, var) for var, coef in dict(InferredResults.mean(axis=1)).items() if coef != 0])
    if intercept: equation = equation.replace('constant', '1')
    equation = equation.replace('+-', '-')
    p = GDExpr.sympy2prefix(GDExpr.parse_expr(equation), 'node', reindex=False, keep_coeff=True)
    pred = GDExpr.eval(p, data, [], strict=False)
    residual = (pred - data[args.target_var])
    result = dict(
        RMSE=np.sqrt(np.mean(residual ** 2)),
        MAE=np.mean(np.abs(residual)),
        MAPE=np.mean(np.abs(residual) / np.abs(data[args.target_var]).clip(1e-6)),
        sMAPE=2 * np.mean(np.abs(residual) / (np.abs(data[args.target_var]) + np.abs(pred))),
        wMAPE=np.sum(np.abs(residual)) / np.sum(np.abs(data[args.target_var])),
        R2=1 - np.mean(residual ** 2) / np.var(data[args.target_var]),
        ACC2=np.mean(np.abs(residual) <= 1e-2),
        ACC3=np.mean(np.abs(residual) <= 1e-3),
        ACC4=np.mean(np.abs(residual) <= 1e-4),
        equation=equation,
        prefix=p
    )

    def is_success(InferredResults, intercept, data):
        """ A heuristic method to automatically determine whether the discovered formula is ground-truth, which requires further manual inspection """
        # select best digits number (0~4)
        ndigits_equations = {}
        for n in range(5):
            equation = '+'.join(["{:.{}f}*{}".format(coef, n, var) for var, coef in dict(InferredResults.mean(axis=1)).items() if np.round(coef, n) != 0])
            if intercept: equation = equation.replace('constant', '1')
            equation = equation.replace('+-', '-')
            ndigits_equations[n] = equation if len(equation) else '0'
        best_result, best_rmse = {}, np.inf
        for eq in ndigits_equations.values():
            p = GDExpr.sympy2prefix(GDExpr.parse_expr(eq), 'node', reindex=False, keep_coeff=True)
            pred = GDExpr.eval(p, data, [], strict=False)
            residual = (pred - data[args.target_var])
            result = dict(
                RMSE=np.sqrt(np.mean(residual ** 2)),
                MAE=np.mean(np.abs(residual)),
                MAPE=np.mean(np.abs(residual) / np.abs(data[args.target_var]).clip(1e-6)),
                sMAPE=2 * np.mean(np.abs(residual) / (np.abs(data[args.target_var]) + np.abs(pred))),
                wMAPE=np.sum(np.abs(residual)) / np.sum(np.abs(data[args.target_var])),
                R2=1 - np.mean(residual ** 2) / np.var(data[args.target_var]),
                ACC2=np.mean(np.abs(residual) <= 1e-2),
                ACC3=np.mean(np.abs(residual) <= 1e-3),
                ACC4=np.mean(np.abs(residual) <= 1e-4)
            )
            if result['RMSE'] >= best_rmse: continue
            best_result = {**result, 'equation': eq, 'prefix': p} #, 'idx': 0, 'ground_truth': GDExpr.prefix2str(prefix)}
            best_rmse = result['RMSE']
        result = best_result
        return (result['ACC4'] > 0.9)
    success = is_success(InferredResults, intercept, data)

    # Log results
    logger.info(f'Discovered equation: {result["equation"]}\n'
                f'RMSE: {result["RMSE"]:.4e}, MAPE: {result["MAPE"]:.4e}, '
                f'wMAPE: {result["wMAPE"]:.4e}, sMAPE: {result["sMAPE"]:.4e}, '
                f'R2: {result["R2"]:.4f}, '
                f'ACC2: {result["ACC2"]:.4f}, ACC3: {result["ACC3"]:.4f}, ACC4: {result["ACC4"]:.4f}')
    
    with open(args.save_path, 'a') as f:
        json.dump(dict(
            host=socket.gethostname(),
            name=args.name, 
            seed=args.seed, 
            time=time.time() - start_time,
            success=str(success),
            RMSE=result['RMSE'],
            MAPE=result['MAPE'],
            wMAPE=result['wMAPE'],
            sMAPE=result['sMAPE'],
            R2=result['R2'],
            ACC2=result['ACC2'],
            ACC3=result['ACC3'],
            ACC4=result['ACC4'],
            result=result['equation'],
            prefix=result['prefix'],
            obs_noise_SNR=args.obs_noise_SNR,
            missing_link_ratio=args.missing_link_ratio,
            spurious_link_ratio=args.spurious_link_ratio,
            library=args.library,
            polynomial_order=args.polynomial_order,
            activation_alpha=args.activation_alpha,
            activation_beta=args.activation_beta,
            activation_gamma=args.activation_gamma,
            coupled_polynomial_order=args.coupled_polynomial_order,
            coupled_activation_alpha=args.coupled_activation_alpha,
            coupled_activation_beta=args.coupled_activation_beta,
            coupled_activation_gamma=args.coupled_activation_gamma,
            func_lib=list(func_lib.columns)
        ), f)
        f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser(description="Run Two-Phase baseline experiment")
    parser.add_argument('--name', type=str, default=f'TwoPhase_{time.strftime("%Y%m%d_%H%M%S")}', help='Experiment name for logging')
    parser.add_argument('--data', type=str, default='./data/synthetic/KUR.json', help='Path to the dataset file')
    parser.add_argument('--vars', type=str, nargs='*', default=['x', 'omega'], help='List of variable names to be used in the model')
    parser.add_argument('--target_var', type=str, default='dx', help='Target variable for the model')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--info_level', choices=['debug', 'info', 'note', 'warning', 'error', 'critical'], default='debug')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--library', type=str, nargs='+', default=['Polynomial', 'Trigonometric', 'Exponential', 
                                                                   'Fractional', 'Activation', 'Rescaling', 
                                                                   'CoupledPolynomial', 'CoupledTrigonometric', 'CoupledExponential', 
                                                                   'CoupledFractional', 'CoupledActivation', 'CoupledRescaling'],
                                                          choices=['Polynomial', 'Trigonometric', 'Exponential', 
                                                                   'Fractional', 'Activation', 'Rescaling',
                                                                   'CoupledPolynomial', 'CoupledTrigonometric', 'CoupledExponential', 
                                                                   'CoupledFractional', 'CoupledActivation', 'CoupledRescaling'],
                                                          help='Function library to use')
    parser.add_argument('--polynomial_order', type=int, default=3, help='Maximal Order of polynomial functions')
    parser.add_argument('--activation_alpha', type=int, nargs='+', default=[1, 5, 10], help='Alpha values for activation functions')
    parser.add_argument('--activation_beta', type=int, nargs='+', default=[0, 1, 5, 10], help='Beta values for activation functions')
    parser.add_argument('--activation_gamma', type=int, nargs='+', default=[1, 2, 5, 10], help='Gamma values for activation functions')
    parser.add_argument('--coupled_polynomial_order', type=int, default=1, help='Maximal Order of coupled polynomial functions')
    parser.add_argument('--coupled_activation_alpha', type=int, nargs='+', default=[1, 5, 10], help='Alpha values for coupled activation functions')
    parser.add_argument('--coupled_activation_beta', type=int, nargs='+', default=[0, 1, 5, 10], help='Beta values for coupled activation functions')
    parser.add_argument('--coupled_activation_gamma', type=int, nargs='+', default=[1, 2, 5, 10], help='Gamma values for coupled activation functions')
    parser.add_argument('--obs_noise_SNR', type=float, default=None, help='Signal-to-Noise Ratio for observation noise.')
    parser.add_argument('--missing_link_ratio', type=float, default=None, help='Fraction of missing links in the network.')
    parser.add_argument('--spurious_link_ratio', type=float, default=None, help='Fraction of spurious links in the network.')
    parser.add_argument('--save_path', type=str, default='./result/baseline_TwoPhase.jsonl')
    
    args, unknown = parser.parse_known_args()
    if unknown: 
        warnings.warn(f'Unknown args: {unknown}')
    init_logger(args.name, f'./log/baseline/{args.name}/info.log', root_name='ND2', info_level=args.info_level)
    setproctitle(f'{args.name}@ZihanYu')
    if args.seed is None: 
        args.seed = np.random.randint(0, 32768)
        seed_all(args.seed)
    logger.info(f'Args: {args}')

    main(args)