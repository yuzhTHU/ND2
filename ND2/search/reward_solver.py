import logging
from copy import deepcopy
import numpy as np
from typing import Tuple, List, Dict, Union
from scipy.optimize import minimize
from ND2.GDExpr import GDExpr


logger = logging.getLogger('ND2.RewardSolver')


class RewardSolver(object):
    """
    1. Fit parameters of a prefix expression to a data
    2. Evaluate the prediction of the prefix expression
    """
    def __init__(self, 
                 Xv:Dict[str, np.ndarray],
                 Xe:Dict[str, np.ndarray],
                 A:np.ndarray,
                 G:np.ndarray,
                 Y:np.ndarray,
                 mask:np.ndarray=None,
                 complexity_base=0.999,
                 sample_num=500,
                 **kwargs):
        self.Xv = {k: np.array(v) for k, v in Xv.items()}
        self.Xe = {k: np.array(v) for k, v in Xe.items()}
        self.A = np.array(A)
        self.G = np.array(G)
        self.Y = np.array(Y)
        self.mask = np.array(mask) if mask is not None else None
        self.complexity_base = complexity_base
        self.sample_num = sample_num
        if kwargs: logger.warning(f'Unused arguments: {kwargs} in RewardSolver')

        self.var_out = self.Y.var() if mask is None else self.Y[self.mask].var()
        self.var_dict = {'A': self.A, 'G': self.G, 'out': self.Y, **self.Xv, **self.Xe}
        assert len(set(Xv.keys()) | set(Xe.keys())) == len(Xv) + len(Xe)

    def solve(self, 
              prefix:List[str], 
              sample=False, 
              x0:Dict[str, List[Union[float, np.ndarray]]]=None, 
              method='L-BFGS-B',
              max_iter=1000,
              **kwargs) -> Tuple[float, Dict[str, List[Union[float, np.ndarray]]]]:
        """
        Arguments:
        - prefix: List[str], the prefix expression to be evaluated
        - sample: bool, whether to sample partial data for calc MSE

        Returns:
        - reward: float, the reward of the prefix expression
        - coef_dict: Dict
            - <C>: List[float], the coefficients of the prefix expression
            - <Cv>: List of (N_Cv, V) np.ndarray, the coefficients of the prefix expression
            - <Ce>: List of (N_Ce, E) np.ndarray, the coefficients of the prefix expression
        """
        if kwargs: logger.warning(f'Unused arguments: {kwargs} in RewardSolver.solve')

        V, E = self.A.shape[0], self.G.shape[0]
        num_C, num_Cv, num_Ce = prefix.count('<C>'), prefix.count('<Cv>'), prefix.count('<Ce>')

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
            return coef_dict
        
        def loss(params):
            coef_dict = params2coefdict(params)
            prefix_with_coef = deepcopy(prefix)
            tmp = {k: list(v) for k, v in coef_dict.items()}
            prefix_with_coef = [tmp[token].pop(0) if token in tmp else token for token in prefix_with_coef]

            pred = GDExpr.eval(prefix_with_coef, var_dict, [], strict=False)
            true = Y
            if self.mask is not None:
                pred = pred[self.mask]
                true = true[self.mask]
            return np.mean((pred - true) ** 2)

        if num_C + num_Cv + num_Ce == 0:
            MSE = loss(np.empty(0))
            coef_dict = {'<C>': [], '<Cv>': [], '<Ce>': []}
        else:
            x0 = x0 or {'<C>': np.random.randn(num_C),
                        '<Cv>': np.random.randn(num_Cv, V),
                        '<Ce>': np.random.randn(num_Ce, E)}
            x0 = np.concatenate([x0['<C>'], x0['<Cv>'].reshape(-1), x0['<Ce>'].reshape(-1)])
            res = minimize(loss, x0, method=method, options={'maxiter': max_iter})
            MSE = res.fun
            coef_dict = params2coefdict(res.x)

        if not np.isfinite(MSE): MSE = np.inf
        r_MSE = MSE / self.var_out.clip(1e-7)
        if r_MSE < 1e-3: reward = 1 - r_MSE
        else: reward = self.complexity_base ** len(prefix) / (1 + r_MSE)
        return reward, coef_dict

    def evaluate(self, prefix:List[str], coef_dict:dict) -> dict:
        """
        Arguments:
        - prefix: List[str], the prefix expression to be evaluated
        - var_dict: dict, the variables dictionary
            - A: (V, V) np.ndarray, the adjacency matrix
            - G: (E, 2) np.ndarray, the edge list
            - out: (T, V) np.ndarray, the output data
            - Cv: (N_Cv, V) np.ndarray, the coefficients of the prefix expression
            - Ce: (N_Ce, E) np.ndarray, the coefficients of the prefix expression
            - other variables: (T, V) / (T, E) / (1, V) / (1, E) np.ndarray
        - coef_list: List[float], the coefficients of the prefix expression

        Returns:
        - result: dict, the evaluation result
            - pred: (T, V) np.ndarray, the prediction of the prefix expression
            - true: (T, V) np.ndarray, the true output data
            - mask: (T, V) np.ndarray, the mask of the output data
            - RMSE: float, the root mean squared error of the prediction
            - MAE: float, the mean absolute error of the prediction
            - MAPE: float, the mean absolute percentage error of the prediction
            - sMAPE: float, the symmetric mean absolute percentage error of the prediction
            - wMAPE: float, the weighted mean absolute percentage error of the prediction
            - R2: float, the R2 score of the prediction
            - ACC2: float, the accuracy of the prediction (1e-2)
            - ACC3: float, the accuracy of the prediction (1e-3)
            - ACC4: float, the accuracy of the prediction (1e-4)
        """

        prefix_with_coef = deepcopy(prefix)
        tmp = {k: list(v) for k, v in coef_dict.items()}
        prefix_with_coef = [tmp[token].pop(0) if token in tmp else token for token in prefix_with_coef]

        pred = GDExpr.eval(prefix_with_coef, self.var_dict, [], strict=False)
        true = self.Y
        residual = (pred - true)
        # result = dict(pred=pred, true=true, mask=self.mask)
        if self.mask is not None:
            true = true[self.mask]
            pred = pred[self.mask]
            residual = residual[self.mask]
        result = dict(
            R2 = 1 - np.mean(residual ** 2) / np.var(true),
            complexity = len(prefix),
            RMSE = np.sqrt(np.mean(residual ** 2)),
            MAE = np.mean(np.abs(residual)),
            MAPE = np.mean(np.abs(residual) / np.abs(true).clip(1e-6)),
            sMAPE = 2 * np.mean(np.abs(residual) / (np.abs(true) + np.abs(pred)).clip(1e-6)),
            wMAPE = np.sum(np.abs(residual)) / np.sum(np.abs(true)),
            ACC2 = np.mean(np.abs(residual) <= 1e-2),
            ACC3 = np.mean(np.abs(residual) <= 1e-3),
            ACC4 = np.mean(np.abs(residual) <= 1e-4),
        )
        return result



class RolloutRewardSolver(RewardSolver):
    """ 专供 population """
    def __init__(self, config, N, name='x'):
        config.sample_num = None
        super().__init__(config)
        self.N = N
        self.name = name
        self.parse = lambda x: x.clip(0, 1)

    def rollout(self, var_dict, coef_list, f, initial, T):
        var_dict[self.name] = initial
        rollout = [var_dict[self.name]]
        for t in range(T - 1):
            for n in range(self.N):
                dt = f(var_dict, coef_list)
                var_dict[self.name] = self.parse(var_dict[self.name] + dt / self.N)
                rollout.append(var_dict[self.name])
        return np.concatenate(rollout, axis=0)

    def get_traj_and_f(self, prefix, var_dict):
        x = var_dict[self.name]
        dx = var_dict['out']
        true = np.concatenate([x, x[(-1,), :] + dx[(-1,), :]], axis=0)
        tmp = var_dict.copy()
        tmp.pop(self.name)
        f = GDExpr.lambdify(prefix, tmp, strict=False)
        if not callable(f): f = lambda *args: f
        return true, f

    def get_MSE_func(self, prefix, var_dict, split_func, sample_num=None):
        true, f = self.get_traj_and_f(prefix, var_dict)
        def MSE_func(params):
            coef_list, coef_dict = split_func(params)
            pred = self.rollout(coef_dict, coef_list, f, true[(0,), :], true.shape[0])[::self.N, :]
            residual = true - pred
            MSE = np.mean(residual ** 2)
            return MSE
        return MSE_func

    def evaluate(self, prefix: List[str], var_dict: dict, coef_list: List[float]) -> dict:
        true, f = self.get_traj_and_f(prefix, var_dict)
        pred = self.rollout(var_dict, coef_list, f, true[(0,), :], true.shape[0])[::self.N, :]
        return self._evaluate(pred, true)

class EcologicalRewardSolver(RewardSolver):
    def __init__(self, config, V_list):
        raise NotImplementedError
        super().__init__(config)
        self.V_list = V_list
    
    def get_MSE_func(self, prefix, var_dict, split_func=None, sample_num=None):
        raise NotImplementedError
        if split_func is None:
            V, E = var_dict['A'].shape[0], var_dict['G'].shape[0]
            N_Coef, N_Cv, N_Ce = prefix.count('<C>'), prefix.count('Cv'), prefix.count('Ce')
            split_func = lambda x: (list(x[:N_Coef]), 
                                    dict(Cv=x[N_Coef:N_Coef + N_Cv * V].reshape(N_Cv, V),
                                        Ce=x[N_Coef + N_Cv * V:].reshape(N_Ce, E)))
        if sample_num is not None:
            N = int(np.ceil(sample_num / var_dict['out'].shape[1]))
            T = var_dict['out'].shape[0]
            if N < T:
                sample_idx = np.random.choice(T, N, replace=False)
                var_dict = var_dict.copy()
                for var in set(var_dict) - {'A', 'G'}:
                    var_dict[var] = var_dict[var][sample_idx]
        f = GDExpr.lambdify(prefix, var_dict)
        
        def MSE_func(params=None, coef_list=None, coef_dict=None):
            if params is not None: 
                coef_list, coef_dict = split_func(params)
            pred = f(coef_dict, coef_list) if callable(f) else f
            residual = var_dict['out'] - pred
            if self.config.use_mask: residual = residual[var_dict['mask']]
            MSE = np.mean(residual ** 2)
            return MSE
        return MSE_func
