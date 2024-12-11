"""
Graph Dynamical Expression
"""
import re
import torch
import logging
import warnings
import numpy as np
import sympy as sp
import networkx as nx
from typing import List, Tuple
from copy import deepcopy
from scipy.optimize import minimize
from .utils import AttrDict
from .sympy_utils import Aggr, Sour, Targ, Rgga, Regular, Sigmoid, LogAbs, SqrtAbs, Abs

logger = logging.getLogger('ND2.GDExpr')
warnings.filterwarnings("ignore", category=RuntimeWarning)

def is_float(value):
    if isinstance(value, float) or isinstance(value, int): return True
    pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    return bool(pattern.match(str(value)))

eps=1e-7
class GDExprClass:
    def __init__(self, config):
        self.config = config
        
        # self.n_words = self.config.model.decoder.n_words
        self.coeff_token = '<C>'
        self.node_coeff_token = '<Cv>'
        self.edge_coeff_token = '<Ce>'

        self.special = self.config.vocabulary.special
        self.placeholder = self.config.vocabulary.placeholder
        self.coefficient = {self.coeff_token: self.config.vocabulary.coefficient}
        self.variable = self.config.vocabulary.variable
        self.operator = self.config.vocabulary.operator
        self.constant = self.config.vocabulary.constant

        self.pad_id = self.special.pad

        self.word2id = self.special | self.placeholder | self.coefficient | self.constant | \
                       self.variable.node | self.variable.edge | \
                       self.operator.binary | self.operator.unary
        self.id2word = {v: k for k, v in self.word2id.items()}
        assert len(self.id2word) ==  len(self.word2id) == len(self.special) + \
                                                          len(self.constant) + \
                                                          len(self.placeholder) + \
                                                          len(self.coefficient) + \
                                                          len(self.variable.node) + \
                                                          len(self.variable.edge) + \
                                                          len(self.operator.binary) + \
                                                          len(self.operator.unary)
    
    def set_config(self, config:AttrDict):
        raise NotImplementedError
        self.config = config
        self.logger.info(f'[GDExpr] Load config from {config}')

        self.n_words = config.model.decoder.n_words
        self.coeff_token = '<C>'

        self.special = config.vocabulary.special
        self.placeholder = config.vocabulary.placeholder
        self.coefficient = {self.coeff_token: config.vocabulary.coefficient}
        self.variable = config.vocabulary.variable
        self.operator = config.vocabulary.operator
        self.constant = config.vocabulary.constant

        self.pad_id = self.special.pad

        self.word2id = self.special | self.placeholder | self.coefficient | self.constant | \
                      self.variable.node | self.variable.edge | self.operator.binary | self.operator.unary
        self.id2word = {v: k for k, v in self.word2id.items()}
        assert len(self.id2word) ==  len(self.word2id) == len(self.special) + \
                                                len(self.constant) + \
                                                len(self.placeholder) + \
                                                len(self.coefficient) + \
                                                len(self.variable.node) + \
                                                len(self.variable.edge) + \
                                                len(self.operator.binary) + \
                                                len(self.operator.unary)

    def parse_expr(self, string:str) -> sp.Expr:
        sympy_expr = sp.parse_expr(string.replace(self.coeff_token, 'Coefficient') \
                                         .replace(self.node_coeff_token, 'Coefficient_node') \
                                         .replace(self.edge_coeff_token, 'Coefficient_edge'),
                                   local_dict={'aggr': Aggr, 'sour': Sour, 'targ': Targ, 'rgga': Rgga,
                                            'sigmoid': Sigmoid, 'regular': Regular,
                                            'logabs': LogAbs, 'sqrtabs': SqrtAbs, 'abs': Abs,
                                            'Coefficient':sp.Symbol(r'\left<C\right>'),
                                            'Coefficient_node': sp.Symbol(r'\left<C_{v}\right>'),
                                            'Coefficient_edge': sp.Symbol(r'\left<C_{e}\right>')})
        return sympy_expr




    def _SGD(self, prefix:List[str], var_dict:dict, coeff_count:int=0):
        if len(prefix) == 0: raise ValueError('Empty prefix')
        item, prefix = prefix[0], prefix[1:]

        if item in self.operator.unary:
            f1, val1, prefix, coeff_count = self._SGD(prefix, var_dict, coeff_count)
        if item in self.operator.binary:
            f1, val1, prefix, coeff_count = self._SGD(prefix, var_dict, coeff_count)
            f2, val2, prefix, coeff_count = self._SGD(prefix, var_dict, coeff_count)

        if item == 'add':
            if f1 is None:
                if f2 is None: return None, val1 + val2, prefix, coeff_count
                else: return lambda coef: val1 + f2(coef), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: f1(coef) + val2, None, prefix, coeff_count
                else: return lambda coef: f1(coef) + f2(coef), None, prefix, coeff_count
        elif item == 'sub':
            if f1 is None:
                if f2 is None: return None, val1 - val2, prefix, coeff_count
                else: return lambda coef: val1 - f2(coef), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: f1(coef) - val2, None, prefix, coeff_count
                else: return lambda coef: f1(coef) - f2(coef), None, prefix, coeff_count
        elif item == 'mul':
            if f1 is None:
                if f2 is None: return None, val1 * val2, prefix, coeff_count
                else: return lambda coef: val1 * f2(coef), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: f1(coef) * val2, None, prefix, coeff_count
                else: return lambda coef: f1(coef) * f2(coef), None, prefix, coeff_count
        elif item == 'div':
            if f1 is None:
                if f2 is None: return None, val1 / val2, prefix, coeff_count
                else: return lambda coef: val1 / f2(coef), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: f1(coef) / val2, None, prefix, coeff_count
                else: return lambda coef: f1(coef) / f2(coef), None, prefix, coeff_count            
        elif item == 'pow':
            if f1 is None:
                if f2 is None: return None, val1 ** val2, prefix, coeff_count
                else: return lambda coef: val1 ** f2(coef), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: f1(coef) ** val2, None, prefix, coeff_count
                else: return lambda coef: f1(coef) ** f2(coef), None, prefix, coeff_count
        elif item == 'rac':
            if f1 is None:
                if f2 is None: return None, val1 ** (1 / val2), prefix, coeff_count
                else: return lambda coef: val1 ** (1 / f2(coef)), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: f1(coef) ** (1 / val2), None, prefix, coeff_count
                else: return lambda coef: f1(coef) ** (1 / f2(coef)), None, prefix, coeff_count
        elif item == 'regular':
            if f1 is None:
                if f2 is None: return None, 1 / (1 + val1 ** (-val2)), prefix, coeff_count
                else: return lambda coef: 1 / (1 + val1 ** (-f2(coef))), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: 1 / (1 + f1(coef) ** (-val2)), None, prefix, coeff_count
                else: return lambda coef: 1 / (1 + f1(coef) ** (-f2(coef))), None, prefix, coeff_count
        elif item == 'exp':
            if f1 is None: return None, val1.exp(), prefix, coeff_count
            else: return lambda coef: f1(coef).exp(), None, prefix, coeff_count
        elif item == 'logabs':
            if f1 is None: return None, val1.abs().log(), prefix, coeff_count
            else: return lambda coef: f1(coef).abs().log(), None, prefix, coeff_count
        elif item == 'sin':
            if f1 is None: return None, val1.sin(), prefix, coeff_count
            else: return lambda coef: f1(coef).sin(), None, prefix, coeff_count
        elif item == 'cos':
            if f1 is None: return None, val1.cos(), prefix, coeff_count
            else: return lambda coef: f1(coef).cos(), None, prefix, coeff_count
        elif item == 'tan':
            if f1 is None: return None, val1.tan(), prefix, coeff_count
            else: return lambda coef: f1(coef).tan(), None, prefix, coeff_count
        elif item == 'abs':
            if f1 is None: return None, val1.abs(), prefix, coeff_count
            else: return lambda coef: f1(coef).abs(), None, prefix, coeff_count
        elif item == 'inv':
            if f1 is None: return None, 1 / val1, prefix, coeff_count
            else: return lambda coef: 1 / f1(coef), None, prefix, coeff_count
        elif item == 'pow2':
            if f1 is None: return None, val1 ** 2, prefix, coeff_count
            else: return lambda coef: f1(coef) ** 2, None, prefix, coeff_count
        elif item == 'pow3':
            if f1 is None: return None, val1 ** 3, prefix, coeff_count
            else: return lambda coef: f1(coef) ** 3, None, prefix, coeff_count
        elif item == 'sqrtabs':
            if f1 is None: return None, val1.abs().sqrt(), prefix, coeff_count
            else: return lambda coef: f1(coef).abs().sqrt(), None, prefix, coeff_count
        elif item == 'neg':
            if f1 is None: return None, -val1, prefix, coeff_count
            else: return lambda coef: -f1(coef), None, prefix, coeff_count
        elif item == 'sinh':
            if f1 is None: return None, val1.sinh(), prefix, coeff_count
            else: return lambda coef: f1(coef).sinh(), None, prefix, coeff_count
        elif item == 'cosh':
            if f1 is None: return None, val1.cosh(), prefix, coeff_count
            else: return lambda coef: f1(coef).cosh(), None, prefix, coeff_count
        elif item == 'tanh':
            if f1 is None: return None, val1.tanh(), prefix, coeff_count
            else: return lambda coef: f1(coef).tanh(), None, prefix, coeff_count
        elif item == 'sigmoid':
            if f1 is None: return None, val1.sigmoid(), prefix, coeff_count
            else: return lambda coef: f1(coef).sigmoid(), None, prefix, coeff_count
        elif item == 'aggr':
            if f1 is None:
                G = torch.from_numpy(var_dict['G']).to(torch.long).cuda()
                E = var_dict['G'].shape[0]
                V = var_dict['A'].shape[0]
                val1 = val1.expand(-1, -1, E).cuda()
                out = torch.zeros((*val1.shape[0:2], V), dtype=val1.dtype, device=val1.device)
                out.scatter_add_(2, G[None, None, :, 1].expand_as(val1), val1)
                return None, out, prefix, coeff_count
            else:
                G = torch.from_numpy(var_dict['G']).to(torch.long).cuda()
                E = var_dict['G'].shape[0]
                V = var_dict['A'].shape[0]
                def aggr(coef):
                    val1 = f1(coef)
                    val1 = val1.expand(-1, -1, E).cuda()
                    out = torch.zeros((*val1.shape[0:2], V), dtype=val1.dtype, device=val1.device)
                    out.scatter_add_(2, G[None, None, :, 1].expand_as(val1), val1)
                    return out
                return aggr, None, prefix, coeff_count
        elif item == 'sour':
            if f1 is None:
                G = torch.from_numpy(var_dict['G']).to(torch.long).cuda()
                if val1.shape[-1] == 1: out = val1.cuda()
                else: out = val1[:, :, G[:, 0]].cuda()
                return None, out, prefix, coeff_count
            else:
                G = torch.from_numpy(var_dict['G']).to(torch.long).cuda()
                def sour(coef):
                    val1 = f1(coef)
                    if val1.shape[-1] == 1: out = val1.cuda()
                    else: out = val1[:, :, G[:, 0]].cuda()
                    return out
                return sour, None, prefix, coeff_count
        elif item == 'targ':
            if f1 is None:
                G = torch.from_numpy(var_dict['G']).to(torch.long).cuda()
                if val1.shape[-1] == 1: out = val1.cuda()
                else: out = val1[:, :, G[:, 1]].cuda()
                return None, out, prefix, coeff_count
            else:
                G = torch.from_numpy(var_dict['G']).to(torch.long).cuda()
                def targ(coef):
                    val1 = f1(coef)
                    if val1.shape[-1] == 1: out = val1.cuda()
                    else: out = val1[:, :, G[:, 1]].cuda()
                    return out
                return targ, None, prefix, coeff_count
        elif item == 'rgga':
            if f1 is None:
                G = torch.from_numpy(var_dict['G']).to(torch.long).cuda()
                E = var_dict['G'].shape[0]
                V = var_dict['A'].shape[0]
                val1 = val1.expand(-1, -1, E).cuda()
                out = torch.zeros((*val1.shape[0:2], V), dtype=val1.dtype, device=val1.device)
                out.scatter_add_(2, G[None, None, :, 0].expand_as(val1), val1)
                return None, out, prefix, coeff_count
            else:
                G = torch.from_numpy(var_dict['G']).to(torch.long).cuda()
                E = var_dict['G'].shape[0]
                V = var_dict['A'].shape[0]
                def rgga(coef):
                    val1 = f1(coef)
                    val1 = val1.expand(-1, -1, E).cuda()
                    out = torch.zeros((*val1.shape[0:2], V), dtype=val1.dtype, device=val1.device)
                    out.scatter_add_(2, G[None, None, :, 0].expand_as(val1), val1)
                    return out
                return rgga, None, prefix, coeff_count
        elif item in self.coefficient:
            # (C, 1, 1)
            return lambda coef: coef[:, coeff_count, None, None], None, prefix, coeff_count + 1
        elif item in self.variable.node | self.variable.edge:
            # (1, N, V) or (1, N, E)
            return None, torch.from_numpy(var_dict[item]).to(torch.float32).unsqueeze(0).cuda(), prefix, coeff_count
        elif item in self.constant:
            # (1, 1, 1)
            return None, torch.tensor(float(eval(item))).view(1, 1, 1).cuda(), prefix, coeff_count
        elif is_float(item):
            # (1, 1, 1)
            return None, torch.tensor(float(item)).view(1, 1, 1).cuda(), prefix, coeff_count
        elif item in self.placeholder:
            raise ValueError(f'Unfilled placeholder {item}')
        else:
            raise ValueError(f'Unknown item {item}')

    def SGD(self, prefix:List[str], var_dict:dict, max_iter=1000, sample_num=500):
        """
        Arguments:
        - prefix: a list of tokens
        - var_dict: Dict[A, G, out, variables...]

        Returns:
        - MSE: float
        - residual: np.ndarray, (N, VorE)
        - best prefix: fill all coefficients with concrete values
        """
        if not self.is_terminal(prefix): return np.nan, None, None
        coeff_num = prefix.count(self.coeff_token)
        if coeff_num == 0:
            pred = self.eval(prefix, var_dict, [])
            residual = var_dict['out'] - pred
            MSE = np.mean(residual**2)
            return MSE, residual, []

        f, _val, _prefix, _count = self._SGD(prefix, var_dict)
        assert _val is None 
        assert len(_prefix) == 0 
        assert _count == coeff_num
        coeff = torch.rand(50, coeff_num) * 20 - 10 # (50, C)
        coeff = coeff.cuda()
        coeff.requires_grad_(True)
        optimizer = torch.optim.Adam([coeff], lr=1e-3)
        min_loss, best_coeff, argmin = np.inf, None, 0
        gt = torch.from_numpy(var_dict['out']).cuda() # (N, VorE)
        for e in range(1000):
            pd = f(coeff).cuda() # (50, N, VorE)
            losses = (pd - gt).square().flatten(-2, -1).mean(-1) # (50,)
            losses.masked_fill_(~losses.isfinite(), np.inf)
            if losses.min() < min_loss:
                min_loss = losses.min().item()
                best_coeff = coeff[losses.argmin()].tolist()
                argmin = e
            if e - argmin > 100: break
            optimizer.zero_grad()
            losses.sum().backward()
            torch.nn.utils.clip_grad_norm_([coeff], 100.)
            optimizer.step()
        pred = self.eval(prefix, var_dict, best_coeff)
        residual = var_dict['out'] - pred
        MSE = np.mean(residual**2)
        return MSE, residual, best_coeff

    def _BFGS(self, prefix:List[str], var_dict:dict, coeff_count:tuple=(0,0,0)):
        if len(prefix) == 0: raise ValueError('Empty prefix')
        item, prefix = prefix[0], prefix[1:]

        if item in self.operator.unary:
            f1, val1, prefix, coeff_count = self._BFGS(prefix, var_dict, coeff_count)
        if item in self.operator.binary:
            f1, val1, prefix, coeff_count = self._BFGS(prefix, var_dict, coeff_count)
            f2, val2, prefix, coeff_count = self._BFGS(prefix, var_dict, coeff_count)

        if item == 'add':
            if f1 is None:
                if f2 is None: return None, np.add(val1, val2), prefix, coeff_count
                else: return lambda coef: np.add(val1, f2(coef)), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: np.add(f1(coef), val2), None, prefix, coeff_count
                else: return lambda coef: np.add(f1(coef), f2(coef)), None, prefix, coeff_count
        elif item == 'sub':
            if f1 is None:
                if f2 is None: return None, np.subtract(val1, val2), prefix, coeff_count
                else: return lambda coef: np.subtract(val1, f2(coef)), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: np.subtract(f1(coef), val2), None, prefix, coeff_count
                else: return lambda coef: np.subtract(f1(coef), f2(coef)), None, prefix, coeff_count
        elif item == 'mul':
            if f1 is None:
                if f2 is None: return None, np.multiply(val1, val2), prefix, coeff_count
                else: return lambda coef: np.multiply(val1, f2(coef)), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: np.multiply(f1(coef), val2), None, prefix, coeff_count
                else: return lambda coef: np.multiply(f1(coef), f2(coef)), None, prefix, coeff_count
        elif item == 'div':
            if f1 is None:
                if f2 is None: return None, np.divide(val1, val2+eps), prefix, coeff_count
                else: return lambda coef: np.divide(val1, f2(coef)+eps), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: np.divide(f1(coef), val2+eps), None, prefix, coeff_count
                else: return lambda coef: np.divide(f1(coef), f2(coef)+eps), None, prefix, coeff_count            
        elif item == 'pow':
            if f1 is None:
                if f2 is None: return None, np.power(val1, val2), prefix, coeff_count
                else: return lambda coef: np.power(val1, f2(coef)), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: np.power(f1(coef), val2), None, prefix, coeff_count
                else: return lambda coef: np.power(f1(coef), f2(coef)), None, prefix, coeff_count
        elif item == 'rac':
            if f1 is None:
                if f2 is None: return None, np.power(val1, np.divide(1, val2+eps)), prefix, coeff_count
                else: return lambda coef: np.power(val1, np.divide(1, f2(coef)+eps)), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: np.power(f1(coef), np.divide(1, val2+eps)), None, prefix, coeff_count
                else: return lambda coef: np.power(f1(coef), np.divide(1, f2(coef)+eps)), None, prefix, coeff_count
        elif item == 'regular':
            if f1 is None:
                if f2 is None: return None, np.divide(1, 1 + np.power(val1, -val2)), prefix, coeff_count
                else: return lambda coef: np.divide(1, 1 + np.power(val1, -f2(coef))), None, prefix, coeff_count
            else:
                if f2 is None: return lambda coef: np.divide(1, 1 + np.power(f1(coef), -val2)), None, prefix, coeff_count
                else: return lambda coef: np.divide(1, 1 + np.power(f1(coef), -f2(coef))), None, prefix, coeff_count
        elif item == 'exp':
            if f1 is None: return None, np.exp(val1), prefix, coeff_count
            else: return lambda coef: np.exp(f1(coef)), None, prefix, coeff_count
        elif item == 'logabs':
            if f1 is None: return None, np.log(np.abs(val1)), prefix, coeff_count
            else: return lambda coef: np.log(np.abs(f1(coef))), None, prefix, coeff_count
        elif item == 'sin':
            if f1 is None: return None, np.sin(val1), prefix, coeff_count
            else: return lambda coef: np.sin(f1(coef)), None, prefix, coeff_count
        elif item == 'cos':
            if f1 is None: return None, np.cos(val1), prefix, coeff_count
            else: return lambda coef: np.cos(f1(coef)), None, prefix, coeff_count
        elif item == 'tan':
            if f1 is None: return None, np.tan(val1), prefix, coeff_count
            else: return lambda coef: np.tan(f1(coef)), None, prefix, coeff_count
        elif item == 'abs':
            if f1 is None: return None, np.abs(val1), prefix, coeff_count
            else: return lambda coef: np.abs(f1(coef)), None, prefix, coeff_count
        elif item == 'inv':
            if f1 is None: return None, np.divide(1, val1+eps), prefix, coeff_count
            else: return lambda coef: np.divide(1, f1(coef)+eps), None, prefix, coeff_count
        elif item == 'pow2':
            if f1 is None: return None, np.power(val1, 2), prefix, coeff_count
            else: return lambda coef: np.power(f1(coef), 2), None, prefix, coeff_count
        elif item == 'pow3':
            if f1 is None: return None, np.power(val1, 3), prefix, coeff_count
            else: return lambda coef: np.power(f1(coef), 3), None, prefix, coeff_count
        elif item == 'sqrtabs':
            if f1 is None: return None, np.sqrt(np.abs(val1)), prefix, coeff_count
            else: return lambda coef: np.sqrt(np.abs(f1(coef))), None, prefix, coeff_count
        elif item == 'neg':
            if f1 is None: return None, -val1, prefix, coeff_count
            else: return lambda coef: -f1(coef), None, prefix, coeff_count
        elif item == 'sinh':
            if f1 is None: return None, np.sinh(val1), prefix, coeff_count
            else: return lambda coef: np.sinh(f1(coef)), None, prefix, coeff_count
        elif item == 'cosh':
            if f1 is None: return None, np.cosh(val1), prefix, coeff_count
            else: return lambda coef: np.cosh(f1(coef)), None, prefix, coeff_count
        elif item == 'tanh':
            if f1 is None: return None, np.tanh(val1), prefix, coeff_count
            else: return lambda coef: np.tanh(f1(coef)), None, prefix, coeff_count
        elif item == 'sigmoid':
            if f1 is None: return None, np.divide(1, 1 + np.exp(-val1)), prefix, coeff_count
            else: return lambda coef: np.divide(1, 1 + np.exp(-f1(coef))), None, prefix, coeff_count
        elif item == 'aggr':
            G = var_dict['G']
            E = var_dict['G'].shape[0]
            V = var_dict['A'].shape[0]
            if f1 is None:
                N = val1.shape[0]
                if val1.shape[1] == 1: val1 = np.repeat(val1, E, axis=1)
                out = np.zeros((val1.shape[0], V))
                if N >= E:
                    for i in range(E):
                        out[:, G[i, 1]] += val1[:, i]
                else:
                    for i in range(N):
                        out[i, :] = np.bincount(G[:, 1], val1[i, :], minlength=V)
                return None, out, prefix, coeff_count
            else:
                def aggr(coef):
                    val1 = f1(coef)
                    N = val1.shape[0]
                    if val1.shape[1] == 1: val1 = np.repeat(val1, E, axis=1)
                    out = np.zeros((val1.shape[0], V))
                    if N >= E:
                        for i in range(E):
                            out[:, G[i, 1]] += val1[:, i]
                    else:
                        for i in range(N):
                            out[i, :] = np.bincount(G[:, 1], val1[i, :], minlength=V)
                    return out
                return aggr, None, prefix, coeff_count
        elif item == 'sour':
            G = var_dict['G']
            if f1 is None:
                if val1.shape[-1] == 1: out = val1
                else: out = val1[:, G[:, 0]]
                return None, out, prefix, coeff_count
            else:
                def sour(coef):
                    val1 = f1(coef)
                    if val1.shape[-1] == 1: out = val1
                    else: out = val1[:, G[:, 0]]
                    return out
                return sour, None, prefix, coeff_count
        elif item == 'targ':
            G = var_dict['G']
            if f1 is None:
                if val1.shape[-1] == 1: out = val1
                else: out = val1[:, G[:, 1]]
                return None, out, prefix, coeff_count
            else:
                def targ(coef):
                    val1 = f1(coef)
                    if val1.shape[-1] == 1: out = val1
                    else: out = val1[:, G[:, 1]]
                    return out
                return targ, None, prefix, coeff_count
        elif item == 'rgga':
            G = var_dict['G']
            E = var_dict['G'].shape[0]
            V = var_dict['A'].shape[0]
            if f1 is None:
                N = val1.shape[0]
                if val1.shape[1] == 1: val1 = np.repeat(val1, E, axis=1)
                out = np.zeros((val1.shape[0], V))
                if N >= E:
                    for i in range(E):
                        out[:, G[i, 0]] += val1[:, i]
                else:
                    for i in range(N):
                        out[i, :] = np.bincount(G[:, 0], val1[i, :], minlength=V)
                return None, out, prefix, coeff_count
            else:
                def rgga(coef):
                    val1 = f1(coef)
                    N = val1.shape[0]
                    if val1.shape[1] == 1: val1 = np.repeat(val1, E, axis=1)
                    out = np.zeros((val1.shape[0], V))
                    if N >= E:
                        for i in range(E):
                            out[:, G[i, 0]] += val1[:, i]
                    else:
                        for i in range(N):
                            out[i, :] = np.bincount(G[:, 0], val1[i, :], minlength=V)
                    return out
                return rgga, None, prefix, coeff_count
        elif item in self.coefficient:
            # (1, 1)
            return lambda coef: coef['<C>'][coeff_count[0], None, None], None, prefix, (coeff_count[0] + 1, coeff_count[1], coeff_count[2])
        elif item in self.variable.node | self.variable.edge:
            # (N, V) or (N, E)
            return None, var_dict[item], prefix, coeff_count
        elif item in self.constant:
            # (1, 1)
            return None, np.array(float(eval(item))).reshape(1, 1), prefix, coeff_count
        elif is_float(item):
            # (1, 1)
            return None, np.array(float(item)).reshape(1, 1), prefix, coeff_count
        elif item == '<Cv>':
            # (1, V)
            return lambda coef: coef['<Cv>'][coeff_count[1], None, :], None, prefix, (coeff_count[0], coeff_count[1] + 1, coeff_count[2])
        elif item == '<Ce>':
            # (1, E)
            return lambda coef: coef['<Ce>'][coeff_count[2], None, :], None, prefix, (coeff_count[0], coeff_count[1], coeff_count[2] + 1)
        elif item in self.placeholder:
            raise ValueError(f'Unfilled placeholder {item}')
        else:
            raise ValueError(f'Unknown item {item}')

    def BFGS(self, prefix:List[str], var_dict:dict, max_iter=1000, sample_num=500, use_mask=False, **kwargs):
        """
        Arguments:
        - prefix: a list of tokens
        - var_dict: Dict[A, G, out, variables...]

        Returns:
        - MSE: float
        - residual: np.ndarray, (N, VorE)
        - best prefix: fill all coefficients with concrete values
        """
        if not self.is_terminal(prefix): return np.nan, None, None
        V = var_dict['A'].shape[0]
        E = var_dict['G'].shape[0]
        N_coef = prefix.count(self.coeff_token)
        N_Cv = prefix.count(self.node_coeff_token)
        N_Ce = prefix.count(self.edge_coeff_token)
        if N_coef + N_Cv + N_Ce == 0:
            pred = self.eval(prefix, var_dict, [])
            residual = var_dict['out'] - pred
            if use_mask: residual = residual[var_dict['mask']]
            MSE = np.mean(residual**2)
            return MSE, residual, []
        
        if sample_num is not None and int(np.ceil(sample_num / var_dict['out'].shape[1])) < var_dict['out'].shape[0]:
            N = int(np.ceil(sample_num / var_dict['out'].shape[1]))
            sample_idx = np.random.choice(var_dict['out'].shape[0], N, replace=False)
            _var_dict = var_dict.copy()
            for var in set(var_dict) - {'A', 'G'}:
                _var_dict[var] = var_dict[var][sample_idx]
            f, _val, _prefix, _count  = self._BFGS(prefix, _var_dict)
            assert _val is None
            assert len(_prefix) == 0
            assert _count == (N_coef, N_Cv, N_Ce)
        else:
            f, _val, _prefix, _count  = self._BFGS(prefix, var_dict)
            assert _val is None
            assert len(_prefix) == 0
            assert _count == (N_coef, N_Cv, N_Ce)
            _var_dict = var_dict
        def MSE_func(x, return_residual=False):
            coeff = x[:N_coef]
            Cv = x[N_coef:N_coef + N_Cv * V].reshape(N_Cv, V)
            Ce = x[N_coef + N_Cv * V:].reshape(N_Ce, E)
            pred = f({'<C>': coeff, '<Cv>': Cv, '<Ce>': Ce})
            residual = _var_dict['out'] - pred
            if use_mask: residual = residual[_var_dict['mask']]
            if return_residual: return np.mean(residual ** 2), residual
            else:               return np.mean(residual ** 2)

        x0 = np.random.randn(N_coef + N_Cv * V + N_Ce * E)
        res = minimize(MSE_func, x0, method='L-BFGS-B', options={'maxiter':max_iter}, **kwargs)

        x = np.array(res.x)
        MSE, residual = MSE_func(x, return_residual=True)
        return MSE, residual, x.tolist()

    def _eval(self, prefix:List[str], var_dict:dict, coef_list:List[float], coeff_count:tuple=(0,0,0), **kwargs):
        if len(prefix) == 0: raise ValueError('Empty prefix')
        item, prefix = prefix[0], prefix[1:]

        if item in self.operator.unary:
            val1, prefix, coeff_count = self._eval(prefix, var_dict, coef_list, coeff_count, **kwargs)
        if item in self.operator.binary:
            val1, prefix, coeff_count = self._eval(prefix, var_dict, coef_list, coeff_count, **kwargs)
            val2, prefix, coeff_count = self._eval(prefix, var_dict, coef_list, coeff_count, **kwargs)

        if item == 'add': return val1 + val2, prefix, coeff_count
        elif item == 'sub': return val1 - val2, prefix, coeff_count
        elif item == 'mul': return val1 * val2, prefix, coeff_count
        elif item == 'div': return np.divide(val1, val2+eps), prefix, coeff_count
        elif item == 'pow': return np.power(val1, val2), prefix, coeff_count
        elif item == 'rac': return np.power(val1, np.divide(1, val2+eps)), prefix, coeff_count
        elif item == 'exp': return np.exp(val1), prefix, coeff_count
        elif item == 'logabs': return np.log(np.abs(val1)), prefix, coeff_count
        elif item == 'sin': return np.sin(val1), prefix, coeff_count
        elif item == 'cos': return np.cos(val1), prefix, coeff_count
        elif item == 'tan': return np.tan(val1), prefix, coeff_count
        elif item == 'abs': return np.abs(val1), prefix, coeff_count
        elif item == 'inv': return np.divide(1, val1+eps), prefix, coeff_count
        elif item == 'pow2': return np.power(val1, 2), prefix, coeff_count
        elif item == 'pow3': return np.power(val1, 3), prefix, coeff_count
        elif item == 'sqrtabs': return np.sqrt(np.abs(val1)), prefix, coeff_count
        elif item == 'neg': return -val1, prefix, coeff_count
        elif item == 'sinh': return np.sinh(val1), prefix, coeff_count
        elif item == 'cosh': return np.cosh(val1), prefix, coeff_count
        elif item == 'tanh': return np.tanh(val1), prefix, coeff_count
        elif item == 'sigmoid': return np.divide(1, 1+np.exp(-val1)), prefix, coeff_count
        elif item == 'regular': return np.divide(1, 1+np.power(val1, -val2)), prefix, coeff_count
        elif item == 'aggr':
            G, A, e = var_dict['G'], var_dict['A'], val1
            V, E = A.shape[0], G.shape[0]
            if not isinstance(e, np.ndarray): e = np.array([[e]])
            if e.shape[-1] == 1: e = np.repeat(e, E, axis=-1)
            v = np.zeros((*e.shape[:-1], V))
            for i in range(e.shape[-1]):
                v[..., G[i, 1]] += e[..., i]
            # if e.shape[0] > e.shape[1]:
            # else:
            #     for i in range(e.shape[0]):
            #         v[i, :] = np.bincount(G[:, 1], e[i, :], minlength=V)

            # G_tensor = torch.from_numpy(G[:, 1]).to(torch.long)
            # e_tensor = torch.from_numpy(e)
            # v_tensor = torch.zeros((e.shape[0], A.shape[0]), dtype=e_tensor.dtype)
            # v_tensor.scatter_add_(1, G_tensor.unsqueeze(0).expand_as(e_tensor), e_tensor)
            # v = v_tensor.numpy()
            return v, prefix, coeff_count
        elif item == 'sour':
            G, A, v = var_dict['G'], var_dict['A'], val1
            V, E = A.shape[0], G.shape[0]
            if not isinstance(v, np.ndarray): v = np.array([[v]])
            if v.shape[-1] == 1: v = np.repeat(v, V, axis=-1)
            return v[..., G[:, 0]], prefix, coeff_count
        elif item == 'targ':
            G, A, v = var_dict['G'], var_dict['A'], val1
            V, E = A.shape[0], G.shape[0]
            if not isinstance(v, np.ndarray): v = np.array([[v]])
            if v.shape[-1] == 1: v = np.repeat(v, V, axis=-1)
            return v[..., G[:, 1]], prefix, coeff_count
        elif item == 'rgga':
            G, A, e = var_dict['G'], var_dict['A'], val1
            V, E = A.shape[0], G.shape[0]
            if not isinstance(e, np.ndarray): e = np.array([[e]])
            if e.shape[-1] == 1: e = np.repeat(e, G.shape[0], axis=-1)
            v = np.zeros((*e.shape[:-1], V))
            for i in range(e.shape[-1]):
                v[..., G[i, 0]] += e[..., i]
            # if e.shape[0] > e.shape[1]:
            # else:
            #     for i in range(e.shape[0]):
            #         v[i, :] = np.bincount(G[:, 0], e[i, :], minlength=V)

            # G_tensor = torch.from_numpy(G[:, 0]).to(torch.long)
            # e_tensor = torch.from_numpy(e)
            # v_tensor = torch.zeros((e.shape[0], A.shape[0]), dtype=e_tensor.dtype)
            # v_tensor.scatter_add_(1, G_tensor.unsqueeze(0).expand_as(e_tensor), e_tensor)
            # v = v_tensor.numpy()
            return v, prefix, coeff_count
        elif item in self.variable.node | self.variable.edge:
            return var_dict[item], prefix, coeff_count
        elif item in self.constant:
            return np.array(float(eval(item))).reshape(1, 1), prefix, coeff_count
        elif is_float(item):
            return np.array(float(item)).reshape(1, 1), prefix, coeff_count
        elif item == self.coeff_token:
            return coef_list[coeff_count[0]], prefix, \
                   (coeff_count[0] + 1, coeff_count[1], coeff_count[2])
        elif item == self.node_coeff_token:
            return var_dict['<Cv>'][coeff_count[1], None, :], prefix, \
                   (coeff_count[0], coeff_count[1] + 1, coeff_count[2])
        elif item == self.edge_coeff_token:
            return var_dict['<Ce>'][coeff_count[2], None, :], prefix, \
                   (coeff_count[0], coeff_count[1], coeff_count[2] + 1)
        elif not kwargs.get('strict', True) and item in var_dict:
            return var_dict[item], prefix, coeff_count
        elif item in self.placeholder:
            raise ValueError(f'Unfilled placeholder {item}')
        else:
            raise ValueError(f'Unknown item {item}')

    def eval(self, prefix:List[str], var_dict:dict, coef_list:List[float], **kwargs):
        """
        - kwargs
            - strict (default: True)
        """
        val, _prefix, _coeff_count = self._eval(prefix, var_dict, coef_list, **kwargs)
        assert len(_prefix) == 0, f'{prefix} -> {_prefix}'
        assert _coeff_count == (prefix.count('<C>'), prefix.count('<Cv>'), prefix.count('<Ce>')), f'{_coeff_count} != ({prefix.count("<C>")}, {prefix.count("<Cv>")}, {prefix.count("<Ce>")})'
        return val
    
    def is_terminal(self, prefix:List[str]):
        for phd in self.placeholder:
            if phd in prefix: return False
        return True

    def get_valid_mask(self, prefix:List[str], var_list:List[str]=None):
        raise DeprecationWarning('Use NN_MCTS.get_mask() instead')
        """
        Arguments:
        - prefix: a list of tokens
        - var_list: a list of usable variables

        invalid: too long prefix / too many coefficients / invalid placeholder type
        """
        max_len = self.config.max_complexity # max length of prefix
        max_coeff_num = self.config.max_coeff_num # max number of coefficients

        first_node = prefix.index('node') if 'node' in prefix else -1
        first_edge = prefix.index('edge') if 'edge' in prefix else -1
        if 0 <= first_node < first_edge: first_edge = -1
        if 0 <= first_edge < first_node: first_node = -1
        # assert max(first_node, first_edge) >= 0
        # if mode == 'first':
        #     if 0 <= first_node < first_edge: first_edge = -1
        #     if 0 <= first_edge < first_node: first_node = -1
        #     assert (first_edge < 0 <= first_node) ^ (first_node < 0 <= first_edge)

        mask = np.zeros((self.n_words,), dtype=bool)
        if max_len - len(prefix) >= 2:
            mask[self.vectorize(list(self.operator.binary))] = True
        if max_len - len(prefix) >= 1:
            mask[self.vectorize(list(self.operator.unary))] = True
        if max_len - len(prefix) >= 0:
            var = list(self.variable.node) if first_node >= 0 else list(self.variable.edge)
            if var_list is not None: var = list(set(var) & set(var_list))
            if len(var): mask[self.vectorize(var)] = True
            
            cnt = 1
            constant_ok = False
            idx = max(first_node, first_edge) - 1
            while cnt > 0 and idx >= 0 and not constant_ok:
                if prefix[idx] in self.operator.binary: cnt -= 2
                elif prefix[idx] in self.operator.unary: cnt -= 1
                elif prefix[idx] in self.placeholder | self.variable.node | self.variable.edge: 
                    cnt += 1; constant_ok = True
                else: cnt += 1
                idx -= 1
            if constant_ok:
                mask[self.vectorize(list(self.constant))] = True
                if prefix.count(self.coeff_token) < max_coeff_num:
                    mask[self.word2id[self.coeff_token]] = True
        if max_len - len(prefix) < 0:
            raise ValueError(f'Invalid max_len {max_len} and prefix {prefix}')
        
        if first_node < 0: mask[self.word2id['aggr']] = mask[self.word2id['rgga']] = False
        if first_edge < 0: mask[self.word2id['sour']] = mask[self.word2id['targ']] = False
        return mask

    def _lambdify(self, prefix:List[str], var_dict, strict=True, 
                  coeff_count:Tuple[int, int, int]=(0, 0, 0)):
        if len(prefix) == 0: raise ValueError('Empty prefix')
        item, prefix = prefix[0], prefix[1:]

        if item in self.operator.unary:
            f1, prefix, coeff_count = self._lambdify(prefix, var_dict, coeff_count=coeff_count, strict=strict)
            c1 = callable(f1)
        if item in self.operator.binary:
            f1, prefix, coeff_count = self._lambdify(prefix, var_dict, coeff_count=coeff_count, strict=strict)
            f2, prefix, coeff_count = self._lambdify(prefix, var_dict, coeff_count=coeff_count, strict=strict)
            c1, c2 = callable(f1), callable(f2)

        if item == 'add': 
            if c1 and c2: return lambda var,coef: f1(var,coef) + f2(var,coef), prefix, coeff_count
            elif c1: return lambda var,coef: f1(var,coef) + f2, prefix, coeff_count
            elif c2: return lambda var,coef: f1 + f2(var,coef), prefix, coeff_count
            else: return f1 + f2, prefix, coeff_count
        elif item == 'sub':
            if c1 and c2: return lambda var,coef: f1(var,coef) - f2(var,coef), prefix, coeff_count
            elif c1: return lambda var,coef: f1(var,coef) - f2, prefix, coeff_count
            elif c2: return lambda var,coef: f1 - f2(var,coef), prefix, coeff_count
            else: return f1 - f2, prefix, coeff_count
        elif item == 'mul':
            if c1 and c2: return lambda var,coef: f1(var,coef) * f2(var,coef), prefix, coeff_count
            elif c1: return lambda var,coef: f1(var,coef) * f2, prefix, coeff_count
            elif c2: return lambda var,coef: f1 * f2(var,coef), prefix, coeff_count
            else: return f1 * f2, prefix, coeff_count
        elif item == 'div':
            if c1 and c2: return lambda var,coef: np.divide(f1(var,coef), f2(var,coef)+eps), prefix, coeff_count
            elif c1: return lambda var,coef: np.divide(f1(var,coef), f2+eps), prefix, coeff_count
            elif c2: return lambda var,coef: np.divide(f1, f2(var,coef)+eps), prefix, coeff_count
            else: return np.divide(f1, f2+eps), prefix, coeff_count
        elif item == 'pow':
            if c1 and c2: return lambda var,coef: np.power(f1(var,coef), f2(var,coef)), prefix, coeff_count
            elif c1: return lambda var,coef: np.power(f1(var,coef), f2), prefix, coeff_count
            elif c2: return lambda var,coef: np.power(f1, f2(var,coef)), prefix, coeff_count
            else: return np.power(f1, f2), prefix, coeff_count
        elif item == 'rac':
            if c1 and c2: return lambda var,coef: np.power(f1(var,coef), np.divide(1, f2(var,coef)+eps)), prefix, coeff_count
            elif c1: return lambda var,coef: np.power(f1(var,coef), np.divide(1, f2+eps)), prefix, coeff_count
            elif c2: return lambda var,coef: np.power(f1, np.divide(1, f2(var,coef)+eps)), prefix, coeff_count
            else: return np.power(f1, np.divide(1, f2+eps)), prefix, coeff_count
        elif item == 'exp':
            if c1: return lambda var,coef: np.exp(f1(var,coef)), prefix, coeff_count
            else: return np.exp(f1), prefix, coeff_count
        elif item == 'logabs':
            if c1: return lambda var,coef: np.log(np.abs(f1(var,coef))), prefix, coeff_count
            else: return np.log(np.abs(f1)), prefix, coeff_count
        elif item == 'sin':
            if c1: return lambda var,coef: np.sin(f1(var,coef)), prefix, coeff_count
            else: return np.sin(f1), prefix, coeff_count
        elif item == 'cos':
            if c1: return lambda var,coef: np.cos(f1(var,coef)), prefix, coeff_count
            else: return np.cos(f1), prefix, coeff_count
        elif item == 'tan':
            if c1: return lambda var,coef: np.tan(f1(var,coef)), prefix, coeff_count
            else: return np.tan(f1), prefix, coeff_count
        elif item == 'abs':
            if c1: return lambda var,coef: np.abs(f1(var,coef)), prefix, coeff_count
            else: return np.abs(f1), prefix, coeff_count
        elif item == 'inv':
            if c1: return lambda var,coef: np.divide(1, f1(var,coef)+eps), prefix, coeff_count
            else: return np.divide(1, f1+eps), prefix, coeff_count
        elif item == 'pow2':
            if c1: return lambda var,coef: np.power(f1(var,coef), 2), prefix, coeff_count
            else: return np.power(f1, 2), prefix, coeff_count
        elif item == 'pow3':
            if c1: return lambda var,coef: np.power(f1(var,coef), 3), prefix, coeff_count
            else: return np.power(f1, 3), prefix, coeff_count
        elif item == 'sqrtabs':
            if c1: return lambda var,coef: np.sqrt(np.abs(f1(var,coef))), prefix, coeff_count
            else: return np.sqrt(np.abs(f1)), prefix, coeff_count
        elif item == 'neg':
            if c1: return lambda var,coef: -f1(var,coef), prefix, coeff_count
            else: return -f1, prefix, coeff_count
        elif item == 'sinh':
            if c1: return lambda var,coef: np.sinh(f1(var,coef)), prefix, coeff_count
            else: return np.sinh(f1), prefix, coeff_count
        elif item == 'cosh':
            if c1: return lambda var,coef: np.cosh(f1(var,coef)), prefix, coeff_count
            else: return np.cosh(f1), prefix, coeff_count
        elif item == 'tanh':
            if c1: return lambda var,coef: np.tanh(f1(var,coef)), prefix, coeff_count
            else: return np.tanh(f1), prefix, coeff_count
        elif item == 'sigmoid':
            if c1: return lambda var,coef: np.divide(1, 1+np.exp(-f1(var,coef))), prefix, coeff_count
            else: return np.divide(1, 1+np.exp(-f1)), prefix, coeff_count
        elif item == 'regular':
            if c1 and c2: return lambda var,coef: np.divide(1, 1+np.power(f1(var,coef), -f2(var,coef))), prefix, coeff_count
            elif c1: return lambda var,coef: np.divide(1, 1+np.power(f1(var,coef), -f2)), prefix, coeff_count
            elif c2: return lambda var,coef: np.divide(1, 1+np.power(f1, -f2(var,coef))), prefix, coeff_count
            else: return np.divide(1, 1+np.power(f1, -f2)), prefix, coeff_count
        elif item == 'aggr':
            G, A = var_dict['G'], var_dict['A']
            V, E = A.shape[0], G.shape[0]
            def my_aggr(e):
                if not isinstance(e, np.ndarray): e = np.array([[e]])
                N = e.shape[0]
                if e.shape[-1] == 1: e = np.repeat(e, E, axis=1)
                out = np.zeros((e.shape[0], V))
                if N >= E: 
                    for i in range(E): out[:, G[i, 1]] += e[:, i]
                else: 
                    for i in range(N): out[i, :] = np.bincount(G[:, 1], e[i, :], minlength=V)
                return out
            if c1: return lambda var,coef: my_aggr(f1(var,coef)), prefix, coeff_count
            else: return my_aggr(f1), prefix, coeff_count
        elif item == 'sour':
            G = var_dict['G']
            def sour(v):
                if not isinstance(v, np.ndarray): v = np.array([[v]])
                if v.shape[-1] == 1: return v
                else: return v[:, G[:, 0]]
            if c1: return lambda var,coef: sour(f1(var,coef)), prefix, coeff_count
            else: return sour(f1), prefix, coeff_count
        elif item == 'targ':
            G = var_dict['G']
            def targ(v):
                if not isinstance(v, np.ndarray): v = np.array([[v]])
                if v.shape[-1] == 1: return v
                else: return v[:, G[:, 1]]
            if c1: return lambda var,coef: targ(f1(var,coef)), prefix, coeff_count
            else: return targ(f1), prefix, coeff_count
        elif item == 'rgga':
            G, A = var_dict['G'], var_dict['A']
            V, E = A.shape[0], G.shape[0]
            def my_aggr(e):
                if not isinstance(e, np.ndarray): e = np.array([[e]])
                N = e.shape[0]
                if e.shape[-1] == 1: e = np.repeat(e, E, axis=1)
                out = np.zeros((e.shape[0], V))
                if N >= E: 
                    for i in range(E): out[:, G[i, 0]] += e[:, i]
                else: 
                    for i in range(N): out[i, :] = np.bincount(G[:, 0], e[i, :], minlength=V)
                return out
            if c1: return lambda var,coef: my_aggr(f1(var,coef)), prefix, coeff_count
            else: return my_aggr(f1), prefix, coeff_count
        elif is_float(item):
            return np.array(float(item)).reshape(1, 1), prefix, coeff_count
        elif item in self.constant:
            return np.array(float(eval(item))).reshape(1, 1), prefix, coeff_count
        elif item == self.coeff_token:
            return lambda var,coef: coef[coeff_count[0]], prefix, \
                   (coeff_count[0]+1, coeff_count[1], coeff_count[2])
        elif item == self.node_coeff_token:
            return lambda var,coef: var['<Cv>'][coeff_count[1], np.newaxis, :], prefix, \
                   (coeff_count[0], coeff_count[1]+1, coeff_count[2])
        elif item == self.edge_coeff_token:
            return lambda var,coef: var['<Ce>'][coeff_count[2], np.newaxis, :], prefix, \
                   (coeff_count[0], coeff_count[1], coeff_count[2]+1)
        elif (item in self.variable.node | self.variable.edge) or (not strict):
            if item in var_dict: return var_dict[item], prefix, coeff_count
            else: return lambda var,coef: var[item], prefix, coeff_count
        elif item in self.placeholder:
            raise ValueError(f'Unfilled placeholder {item}')
        else:
            raise ValueError(f'Unknown item {item}')

        # elif item == 'sub': return lambda var,coef: f1(var,coef) - f2(var,coef), prefix, coeff_count
        # elif item == 'mul': return lambda var,coef: f1(var,coef) * f2(var,coef), prefix, coeff_count
        # elif item == 'div': return lambda var,coef: np.divide(f1(var,coef), f2(var,coef)), prefix, coeff_count
        # elif item == 'pow': return lambda var,coef: np.power(f1(var,coef), f2(var,coef)), prefix, coeff_count
        # elif item == 'rac': return lambda var,coef: np.power(f1(var,coef), np.divide(1, f2(var,coef))), prefix, coeff_count
        # elif item == 'exp': return lambda var,coef: np.exp(f1(var,coef)), prefix, coeff_count
        # elif item == 'logabs': return lambda var,coef: np.log(np.abs(f1(var,coef))), prefix, coeff_count
        # elif item == 'sin': return lambda var,coef: np.sin(f1(var,coef)), prefix, coeff_count
        # elif item == 'cos': return lambda var,coef: np.cos(f1(var,coef)), prefix, coeff_count
        # elif item == 'tan': return lambda var,coef: np.tan(f1(var,coef)), prefix, coeff_count
        # elif item == 'abs': return lambda var,coef: np.abs(f1(var,coef)), prefix, coeff_count
        # elif item == 'inv': return lambda var,coef: np.divide(1, f1(var,coef)), prefix, coeff_count
        # elif item == 'pow2': return lambda var,coef: np.power(f1(var,coef), 2), prefix, coeff_count
        # elif item == 'pow3': return lambda var,coef: np.power(f1(var,coef), 3), prefix, coeff_count
        # elif item == 'sqrtabs': return lambda var,coef: np.sqrt(np.abs(f1(var,coef))), prefix, coeff_count
        # elif item == 'neg': return lambda var,coef: -f1(var,coef), prefix, coeff_count
        # elif item == 'sinh': return lambda var,coef: np.sinh(f1(var,coef)), prefix, coeff_count
        # elif item == 'cosh': return lambda var,coef: np.cosh(f1(var,coef)), prefix, coeff_count
        # elif item == 'tanh': return lambda var,coef: np.tanh(f1(var,coef)), prefix, coeff_count
        # elif item == 'sigmoid': return lambda var,coef: np.divide(1, 1+np.exp(-f1(var,coef))), prefix, coeff_count
        # elif item == 'regular': return lambda var,coef: np.divide(1, 1+np.power(f1(var,coef), -f2(var,coef))), prefix, coeff_count
        # elif item == 'aggr':
        #     def aggr(var,coef):
        #         e = f1(var,coef)
        #         G, A = var['G'], var['A']
        #         if not isinstance(e, np.ndarray):
        #             e = np.full((1, G.shape[0]), e)
        #         out = np.zeros((e.shape[0], A.shape[0]))
        #         if e.shape[0] > e.shape[1]:
        #             for i in range(e.shape[1]):
        #                 out[:, G[i, 1]] += e[:, i]
        #         else:
        #             for i in range(e.shape[0]):
        #                 out[i, :] = np.bincount(G[:, 1], e[i, :], minlength=A.shape[0])
        #         # G_tensor = torch.from_numpy(G[:, 1]).to(torch.long)
        #         # e_tensor = torch.from_numpy(e)
        #         # out_tensor = torch.zeros((e.shape[0], A.shape[0]), dtype=e_tensor.dtype)
        #         # out_tensor.scatter_add_(1, G_tensor.unsqueeze(0).expand_as(e_tensor), e_tensor)
        #         # out = out_tensor.numpy()
        #         return out
        #     return aggr, prefix, coeff_count
        # elif item == 'sour':
        #     def sour(var,coef):
        #         v = f1(var,coef)
        #         G, A = var['G'], var['A']
        #         if not isinstance(v, np.ndarray):
        #             v = np.full((1, A.shape[0]), v)
        #         return v[:, G[:, 0]]
        #     return sour, prefix, coeff_count
        # elif item == 'targ':
        #     def sour(var,coef):
        #         v = f1(var,coef)
        #         G, A = var['G'], var['A']
        #         if not isinstance(v, np.ndarray):
        #             v = np.full((1, A.shape[0]), v)
        #         return v[:, G[:, 1]]
        #     return sour, prefix, coeff_count
        # elif item in self.variable.node | self.variable.edge:
        #     return lambda var,coef: var[item], prefix, coeff_count
        # elif item in self.coefficient:
        #     return lambda var,coef: coef[coeff_count], prefix, coeff_count + 1
        # elif item in self.constant:
        #     return lambda var,coef: float(eval(item)), prefix, coeff_count
        # elif is_float(item):
        #     return lambda var,coef: float(item), prefix, coeff_count
        # elif item in self.placeholder:
        #     raise ValueError(f'Unfilled placeholder {item}')
        # else:
        #     raise ValueError(f'Unknown item {item}')
    
    def lambdify(self, prefix:List[str], var_dict:dict, strict=True):
        """
        Returns:
        - f: A callable function or a np.ndarray value
          params (when f is a callable function): 
            - var_dict
            - coef_list
        """
        # def aggr(e, G, A):
        #     if e.shape[1] == 1: e = np.repeat(e, G.shape[0], axis=1)
        #     out = np.zeros((e.shape[0], A.shape[0]))
        #     if e.shape[0] > e.shape[1]:
        #         for i in range(e.shape[1]):
        #             out[:, G[i, 1]] += e[:, i]
        #     else:
        #         for i in range(e.shape[0]):
        #             out[i, :] = np.bincount(G[:, 1], e[i, :], minlength=A.shape[0])
        #     return out

        # def sour(v, G, A):
        #     if v.shape[-1] == 1: return v
        #     else: return v[:, G[:, 0]]
        
        # def targ(v, G, A):
        #     if v.shape[-1] == 1: return v
        #     else: return v[:, G[:, 1]]

        # prefix = deepcopy(prefix)
        # cnt = [0, 0, 0]
        # for idx, symbol in enumerate(prefix):
        #     if symbol == self.coeff_token: prefix[idx] = f'coef_list[{cnt[0]}]'; cnt[0] += 1
        #     if symbol == self.node_coeff_token: prefix[idx] = f'Cv[:, {cnt[1]}]'; cnt[1] += 1
        #     if symbol == self.edge_coeff_token: prefix[idx] = f'Ce[:, {cnt[2]}]'; cnt[2] += 1

        # expr = self.prefix2str(prefix)
        # global_dict = {'exp': np.exp,
        #                'logabs': lambda x: np.log(np.abs(x)), 
        #                'sin': np.sin, 
        #                'cos': np.cos,
        #                'tan': np.tan,
        #                'abs': np.abs,
        #                'sqrtabs': lambda x: np.sqrt(np.abs(x)), 
        #                'sinh': np.sinh,
        #                'cosh': np.cosh,
        #                'tanh': np.tanh,
        #                'sigmoid': lambda x: np.divide(1, 1+np.exp(-x)),
        #                'regular': lambda x, y: np.divide(1, 1+np.power(x, -y)),
        #                'aggr': aggr, 
        #                'sour': sour, 
        #                'targ': targ,
        #                **var_dict}
        # f = lambda coef_dict, coef_list: eval(expr, {**global_dict, **coef_dict, 'coef_list': coef_list})
    
        f, _prefix, _coeff_count = self._lambdify(prefix, var_dict, strict=strict)
        assert len(_prefix) == 0, f'{prefix} -> {_prefix}'
        assert _coeff_count == (prefix.count(self.coeff_token),
                                prefix.count(self.node_coeff_token),
                                prefix.count(self.edge_coeff_token)), _coeff_count
        return f
    
    def parse_float(self, data:np.ndarray, num_e_bits=5, num_m_bits=10):
        data[np.isnan(data)] = 0.0
        data.clip(-2**(2**(num_e_bits-1)), 2**(2**(num_e_bits-1)), out=data)
        # SIGN BIT
        signs = np.where(data>=0, 0, 1)[..., np.newaxis] # 0 is plus and 1 is minus
        data = np.abs(data)
        ## EXPONENT BIT
        e_scientific = np.floor(np.log2(data)).clip(-2**(num_e_bits-1) + 1, 2**(num_e_bits-1))
        e_decimal = e_scientific + 2**(num_e_bits-1) - 1
        exponents = np.zeros((*e_decimal.shape, num_e_bits))
        for i in range(num_e_bits):
            exponents[..., num_e_bits-i-1] = e_decimal % 2
            e_decimal //= 2
        ## MANTISSA
        data = (data / 2**e_scientific) % 1
        mantissas = np.zeros((*data.shape, num_m_bits))
        for i in range(num_m_bits):
            data *= 2
            mantissas[..., i] = np.floor(data)
            data %= 1
        return np.concatenate([signs, exponents, mantissas], axis=-1)

    def vectorize(self, prefix:List[str]):
        return np.vectorize(self.word2id.get)(prefix)

    def _analysis_parent(self, prefix, parent_idx, cur_idx):
        if len(prefix) == 0: raise ValueError('Empty prefix')
        item, prefix = prefix[0], prefix[1:]
        if item in self.operator.unary:
            parent1, prefix = self._analysis_parent(prefix, cur_idx, cur_idx+1)
            return [parent_idx] + parent1, prefix
        elif item in self.operator.binary:
            parent1, prefix = self._analysis_parent(prefix, cur_idx, cur_idx+1)
            parent2, prefix = self._analysis_parent(prefix, cur_idx, cur_idx+len(parent1)+1)
            return [parent_idx] + parent1 + parent2, prefix
        else: # placeholder / variable / constant / coefficient
            return [parent_idx], prefix

    def analysis_parent(self, prefix:List[str], sentinel_idx=None, start_from=0):
        """
        Arguments:
        - prefix: a list of tokens
        - sentinel_idx: the parent index of first token
        - start_from: the index of first token

        Examples:
            prefix = ['add', 'sin', 'v1', 'v2'], sentinel_idx = None, start_from = 0
                -> parents = [None, 0, 1, 0]
            prefix = ['add', 'sin', 'v1', 'v2'], sentinel_idx = 0, start_from = 1
                -> parents = [0, 1, 2, 1]
        """
        parents, _prefix = self._analysis_parent(prefix, sentinel_idx, start_from)
        assert len(_prefix) == 0, f'{prefix} -> {_prefix}'
        return parents

    def _analysis_type(self, prefix, root_type):
        if len(prefix) == 0: raise ValueError('Empty prefix')
        item, prefix = prefix[0], prefix[1:]
        if item in ['aggr', 'rgga']:
            assert root_type == 'node', root_type
            type1, prefix = self._analysis_type(prefix, 'edge')
            return [root_type, *type1], prefix
        elif item in ['sour', 'targ']:
            assert root_type == 'edge', root_type
            type1, prefix = self._analysis_type(prefix, 'node')
            return [root_type, *type1], prefix
        elif item in self.operator.unary:
            type1, prefix = self._analysis_type(prefix, root_type)
            return [root_type, *type1], prefix
        elif item in self.operator.binary:
            type1, prefix = self._analysis_type(prefix, root_type)
            type2, prefix = self._analysis_type(prefix, root_type)
            return [root_type, *type1, *type2], prefix
        elif item in self.placeholder:
            assert item == root_type, item
            return [root_type], prefix
        else: # variable / constant / coefficient
            return [root_type], prefix

    def analysis_type(self, prefix:List[str], root_type:str):
        assert root_type in self.placeholder
        types, _prefix = self._analysis_type(prefix, root_type)
        assert len(_prefix) == 0, f'{prefix} -> {_prefix}'
        return types

    def _prefix2str(self, prefix, N:int=4):
        if len(prefix) == 0: raise ValueError('Empty prefix')
        item, prefix = prefix[0], prefix[1:]
        
        if item in self.operator.unary:
            arg1, prefix = self._prefix2str(prefix)
        if item in self.operator.binary:
            arg1, prefix = self._prefix2str(prefix)
            arg2, prefix = self._prefix2str(prefix)
        if item == 'add': return f'({arg1}+{arg2})', prefix
        elif item == 'sub': return f'({arg1}-{arg2})', prefix
        elif item == 'mul': return f'({arg1}*{arg2})', prefix
        elif item == 'div': return f'({arg1}/{arg2})', prefix
        elif item == 'pow': return f'({arg1}**{arg2})', prefix
        elif item == 'rac': return f'({arg1}**(1/{arg2}))', prefix
        elif item == 'exp': return f'exp({arg1})', prefix
        elif item == 'logabs': return f'logabs({arg1})', prefix
        elif item == 'sin': return f'sin({arg1})', prefix
        elif item == 'cos': return f'cos({arg1})', prefix
        elif item == 'tan': return f'tan({arg1})', prefix
        elif item == 'abs': return f'abs({arg1})', prefix
        elif item == 'inv': return f'(1/{arg1})', prefix
        elif item == 'pow2': return f'({arg1}**2)', prefix
        elif item == 'pow3': return f'({arg1}**3)', prefix
        elif item == 'sqrtabs': return f'sqrtabs({arg1})', prefix
        elif item == 'neg': return f'(-{arg1})', prefix
        elif item == 'sinh': return f'sinh({arg1})', prefix
        elif item == 'cosh': return f'cosh({arg1})', prefix
        elif item == 'tanh': return f'tanh({arg1})', prefix
        elif item == 'sigmoid': return f'sigmoid({arg1})', prefix
        elif item == 'regular': return f'regular({arg1}, {arg2})', prefix 
            # return f'({arg1}**{arg2})/({arg1}**{arg2}+1)', prefix
        elif item == 'aggr': return f'aggr({arg1})', prefix
        elif item == 'rgga': return f'rgga({arg1})', prefix
        elif item == 'sour': return f'sour({arg1})', prefix
        elif item == 'targ': return f'targ({arg1})', prefix
        elif item in self.word2id: return item, prefix # placeholder / variable / constant / coefficient
        elif item in ['<Cv>', '<Ce>']: return item, prefix
        elif is_float(item): return format(float(item), f'.{N}f') if '.' in str(item) else str(item), prefix
        elif isinstance(item, np.ndarray): return f'<{np.mean(item)}+-{np.std(item)} ({len(item.reshape(-1))})>', prefix
        else:
            # raise ValueError(f'Unknown item {item}')
            # logger.warning(f'Unknown item {item}')
            return item, prefix

    def prefix2str(self, prefix:List[str]) -> str:
        string, _prefix = self._prefix2str(prefix)
        assert len(_prefix) == 0, f'{prefix} -> {_prefix}'
        return string

    def _sympy2prefix(self, sympy_expr, **kwargs):
        item = sympy_expr.func.__name__.lower()
        if item == 'add':
            sub_args = [-arg for arg in sympy_expr.args if arg.as_coeff_mul()[0] < 0]
            plus_args = [arg for arg in sympy_expr.args if arg.as_coeff_mul()[0] >= 0]
            prefix = []
            if len(sub_args): prefix.append('sub' if len(plus_args) else 'neg')
            prefix.extend(['add'] * (len(plus_args) - 1))
            for arg in plus_args:
                prefix.extend(self._sympy2prefix(arg, **kwargs))
            prefix.extend(['add'] * (len(sub_args) - 1))
            for arg in sub_args:
                prefix.extend(self._sympy2prefix(arg, **kwargs))
            return prefix
        elif item == 'mul':
            try:
                den_args = [1/arg for arg in sympy_expr.args if arg.is_Pow and arg.exp < 0]
                num_args = [arg for arg in sympy_expr.args if not arg.is_Pow or arg.exp >= 0]
            except TypeError as e:
                den_args = []
                num_args = list(sympy_expr.args)
            prefix = []
            if len(den_args): prefix.append('div' if len(num_args) else 'inv')
            prefix.extend(['mul'] * (len(num_args) - 1))
            for arg in num_args:
                prefix.extend(self._sympy2prefix(arg, **kwargs))
            prefix.extend(['mul'] * (len(den_args) - 1))
            for arg in den_args:
                prefix.extend(self._sympy2prefix(arg, **kwargs))
            return prefix
        elif item == 'pow' and sympy_expr.exp == 3 and 'pow3' in self.word2id:
            return ['pow3'] + self._sympy2prefix(sympy_expr.base, **kwargs)
        elif item == 'pow' and sympy_expr.exp == 2 and 'pow2' in self.word2id:
            return ['pow2'] + self._sympy2prefix(sympy_expr.base, **kwargs)
        elif item == 'pow' and sympy_expr.exp == -1 and 'inv' in self.word2id:
            return ['inv'] + self._sympy2prefix(sympy_expr.base, **kwargs)
        elif item == 'pow' and sympy_expr.exp == 0.5 and 'sqrt' in self.word2id:
            return ['sqrt'] + self._sympy2prefix(sympy_expr.base, **kwargs)
        elif item == 'pow' and str(1/sympy_expr.exp) in self.constant and 'rac' in self.word2id:
            return ['rac'] + self._sympy2prefix(sympy_expr.base, **kwargs) + [str(1/sympy_expr.exp)]
        elif item == 'exp':
            return ['exp'] + self._sympy2prefix(sympy_expr.args[0], **kwargs)
        elif item == 'tan':
            return ['tan'] + self._sympy2prefix(sympy_expr.args[0], **kwargs)
        elif item in self.operator.binary:
            assert len(sympy_expr.args) == 2
            prefix = [item]
            prefix.extend(self._sympy2prefix(sympy_expr.args[0], **kwargs))
            prefix.extend(self._sympy2prefix(sympy_expr.args[1], **kwargs))
            return prefix
        elif item in self.operator.unary:
            assert len(sympy_expr.args) == 1
            return [item] + self._sympy2prefix(sympy_expr.args[0], **kwargs)
        elif item == 'float':
            return [str(sympy_expr)] if kwargs.get('keep_coeff', False) else [self.coeff_token]
        elif item == 'symbol':
            return [sympy_expr.name]
        elif sympy_expr.is_constant():
            if str(sympy_expr) in self.constant:
                return [str(sympy_expr)]
            elif str(-sympy_expr) in self.constant:
                return ['neg', str(-sympy_expr)]
            else:
                return [float(sympy_expr)] if kwargs.get('keep_coeff', False) else [self.coeff_token]
        elif item == 're':
            return self._sympy2prefix(sympy_expr.args[0], **kwargs)
        else:
            raise ValueError(f'Unknown item {item}')
    
    def sympy2prefix(self, sympy_expr:sp.Expr, root_type:str, **kwargs):
        """
        - kwargs: 
            - reindex (default: True) whether to reindex variables
            - keep_coeff (default: False) whether to keep coefficient as float
        """
        assert root_type in self.placeholder
        prefix = self._sympy2prefix(sympy_expr, **kwargs)
        for idx, item in enumerate(prefix):
            if item == r'\left<C\right>': # same as in the self.parse_expr
                prefix[idx] = self.coeff_token
            elif item == r'\left<C_{v}\right>':
                prefix[idx] = '<Cv>'
            elif item == r'\left<C_{e}\right>':
                prefix[idx] = '<Ce>'

        if not kwargs.get('reindex', True): return prefix 
        # reindex variables
        var_names = set(map(str,sympy_expr.free_symbols)) - {self.coeff_token}
        tmp_map = {str(var):self.coeff_token for var in var_names}
        tmp_prefix = [tmp_map.get(item, item) for item in prefix]
        types = self.analysis_type(tmp_prefix, root_type)
    
        reindex_map = {}
        var = {phd: list(self.variable[phd]) for phd in self.placeholder}
        for item, type in zip(prefix, types):
            if item not in var_names: continue
            if item in reindex_map: 
                assert reindex_map[item] in self.variable[type]
            else:
                reindex_map[item] = var[type].pop(0)
        prefix = [reindex_map.get(item, item) for item in prefix]
        return prefix, reindex_map

    def _random_fill_expr(self, total_len, root_type, **hints):
        if total_len == 0:
            raise ValueError()
        elif total_len == 1:
            prob_type = [hints[i] for i in ['var', 'const', 'coeff']]
            if hints.get('force_var', True): prob_type = [1, 0, 0]
            p = np.array(prob_type, dtype=np.float64) / sum(prob_type)
            item_type = np.random.choice(3, p=p)
            if item_type == 0:
                var = list(self.variable[root_type])
                prob_var = [hints.get(f'var_{v}', 1) for v in var]
                if sum(prob_var) == 0:
                    logger.debug(f'No chooseable variable for "{root_type}" type')
                    return [self.coeff_token]
                p = np.array(prob_var, dtype=np.float64) / sum(prob_var)
                return [np.random.choice(var, p=p)]
            elif item_type == 1:
                return [np.random.choice(list(self.constant))]
            else: # item_type == 2
                return [self.coeff_token]
        elif total_len == 2:
            for op in self.operator.binary:
                hints[f'op_{op}'] = 0

        operator = list(self.operator.unary | self.operator.binary)
        prob_op = [hints.get(f'op_{op}', 1) for op in operator]
        if root_type == 'node': 
            prob_op[operator.index('sour')] = 0
            prob_op[operator.index('targ')] = 0
        elif root_type == 'edge':
            prob_op[operator.index('aggr')] = 0
            prob_op[operator.index('rgga')] = 0
        p = np.array(prob_op, dtype=np.float64) / sum(prob_op)
        op = np.random.choice(operator, p=p)

        if op in ['pow', 'rac', 'regular']:
            # the second agrument must be a const
            hints['force_var'] = True
            left = self._random_fill_expr(total_len - 1 - 1, root_type, **hints)
            hints['force_var'] = False
            hints['var'] = 0
            hints['coeff'] = 0
            hints['const'] = 1
            right = self._random_fill_expr(1, root_type, **hints)
            return [op] + left + right
        elif op in self.operator.binary:
            split = np.random.randint(1, total_len-1)
            hints['force_var'] = False
            left = self._random_fill_expr(split, root_type, **hints)
            hints['force_var'] = not any([var in left for phd in self.placeholder 
                                                      for var in self.variable[phd]])
            right = self._random_fill_expr(total_len - 1 - split, root_type, **hints)
            return [op] + left + right
        elif op in ['sin', 'cos', 'tan']:
            # cannot be nested
            for _op in ['sin', 'cos', 'tan']: hints[f'prob_op_{_op}'] = 0
            hints['force_var'] = True
            child = self._random_fill_expr(total_len - 1, root_type, **hints)
            return [op] + child
        elif op in ['exp', 'logabs', 'rac', 'tanh', 'sigmoid']:
            # cannot be self-nested
            hints[f'prob_op_{op}'] = 0
            hints['force_var'] = True
            child = self._random_fill_expr(total_len - 1, root_type, **hints)
            return [op] + child
        elif op in ['sour', 'targ']:
            # change root_type to 'node'
            hints['force_var'] = True
            child = self._random_fill_expr(total_len - 1, 'node', **hints)
            return [op] + child
        elif op in ['aggr', 'rgga']:
            # change root_type to 'edge'
            hints['force_var'] = True
            child = self._random_fill_expr(total_len - 1, 'edge', **hints)
            return [op] + child
        elif op in self.operator.unary:
            hints['force_var'] = True
            child = self._random_fill_expr(total_len - 1, root_type, **hints)
            return [op] + child
        else:
            raise ValueError(f'Unknown operator {op}')

    def random_fill_expr(self, total_len:int, prefix:List[str], **probs):
        """
        Arguments:
        - total_len: the length of the expression
        - prefix: a list of words
        - reindex: whether to reindex the variables
        - probs: the probability of different operators and variables
        e.g. total_len=9, prefix=[add, <V>, aggr, <E>], then return [add, v1, aggr, sin, sub, sour, v2, targ, v2]
        """
        if len(prefix) > total_len:
            raise ValueError('Invalid root prefix')
        
        phd_pos = [pos for pos, item in enumerate(prefix) if item in self.placeholder]
        n_to_fill = total_len - (len(prefix) - len(phd_pos))
        splits = np.random.choice(np.arange(1, n_to_fill), len(phd_pos)-1, replace=False)
        splits = [0] + np.sort(splits).tolist() + [n_to_fill]
        phd_len = np.diff(splits)

        for idx in range(len(phd_pos)-1, -1, -1):
            pos = phd_pos[idx]
            phd = prefix[pos]
            lng = phd_len[idx]
            hints = {
                # Var or Coeff or Const
                'var': probs.get('var', 60),
                'coeff': probs.get('coeff', 30),
                'const': probs.get('const', 10),
                # Different Operators (default: 1)
                'op_add': probs.get('op_add', 20),
                'op_sub': probs.get('op_sub', 20),
                'op_mul': probs.get('op_mul', 30),
                'op_div': probs.get('op_div', 2),
                'op_sin': probs.get('op_sin', 3),
                'op_pow2': probs.get('op_pow2', 2),
                'op_pow3': probs.get('op_pow3', 2),
                'op_sigmoid': probs.get('op_sigmoid', 2),
                'op_regular': probs.get('op_regular', 2),
                'op_aggr': probs.get('op_aggr', 10),
                'op_sour': probs.get('op_sour', 10),
                'op_term': probs.get('op_term', 10),
                # Different Variables
                **{f'var_{var}': 1 
                   for phd in self.placeholder 
                   for var in self.variable[phd]},
            }
            arg = self._random_fill_expr(lng, phd, **hints)
            prefix = prefix[:pos] + arg + prefix[pos+1:]

        return prefix
    
    def decompose(self, prefix:List[str], root_type:str, choose='final'):
        """
        Arguments:
        - prefix: a list of words
        - root_type: 'node' or 'edge'
        - choose: 'final', 'random', or 'first'
        
        Returns:
        - prefix: a list of words
        - policy: the decomposed operator
        - index: the position of the decomposed operator

        e.g., prefix = ['add', 'add', 'node', 'node', 'node'], return (['add', 'node', 'node'], add, 1)
        """
        if len(prefix) == 0: raise ValueError('Empty prefix')
        assert root_type in self.placeholder
        assert choose in ['final', 'random', 'first']

        if self.config.decomposer.use_random_index and choose != 'random':
            logger.warning(f'Since you are using random index, the choose method will be changed to random')
            choose = 'random'

        leaf_pos = []
        for i, item in enumerate(prefix):
            assert item in self.word2id, item
            if item in self.placeholder: continue
            if item in self.operator.unary:
                if prefix[i+1] in self.placeholder: leaf_pos.append(i)
            elif item in self.operator.binary:
                if prefix[i+1] in self.placeholder and \
                   prefix[i+2] in self.placeholder: leaf_pos.append(i)
            else: # variables / constants / coefficient
                leaf_pos.append(i)
            
        if len(leaf_pos) == 0:
            raise ValueError(f'No leaf in {prefix}')
        if choose == 'random':
            idx = np.random.choice(leaf_pos)
        elif choose == 'first':
            idx = leaf_pos[0]
        elif choose == 'final':
            idx = leaf_pos[-1]

        if prefix[idx] in self.operator.binary:
            skip = 2
        elif prefix[idx] in self.operator.unary:
            skip = 1
        else: # variables / constants / coefficient
            skip = 0
        phd = self.analysis_type(prefix, root_type)[idx]
        return [*prefix[:idx], phd, *prefix[idx+1+skip:]], prefix[idx], idx
    
    def find_first_placeholder(self, prefix:List[str]):
        for index, item in enumerate(prefix):
            if item in self.placeholder: return index
        else: 
            return -1 # no placeholder

    def evaluate(self, prefix:List[str], var_dict, coef_list, use_mask=False):
        # raise DeprecationWarning('Please use RewardSolver.evaluate instead')
        pred = self.eval(prefix, var_dict, coef_list)
        residual = (pred - var_dict['out'])
        if use_mask: residual = residual[var_dict['mask']]
        result = dict(
            RMSE = np.sqrt(np.mean(residual ** 2)),
            MAE = np.mean(np.abs(residual)),
            MAPE = np.mean(np.abs(residual) / np.abs(var_dict['out']).clip(1e-6)),
            sMAPE = 2 * np.mean(np.abs(residual) / (np.abs(var_dict['out']) + np.abs(pred))),
            wMAPE = np.sum(np.abs(residual)) / np.sum(np.abs(var_dict['out'])),
            R2 = 1 - np.mean(residual ** 2) / np.var(var_dict['out']),
            ACC2 = np.mean(np.abs(residual) <= 1e-2),
            ACC3 = np.mean(np.abs(residual) <= 1e-3),
            ACC4 = np.mean(np.abs(residual) <= 1e-4)
        )
        return result

    def count_coef(self, prefix:List[str], V=1, E=1):
        return prefix.count(self.coeff_token) + \
               prefix.count(self.node_coeff_token) * V + \
               prefix.count(self.edge_coeff_token) * E


GDExpr = GDExprClass(AttrDict.load_yaml_str("""
decomposer:
    use_random_index: False
vocabulary:
    special:
        pad: 0
        sos: 1
        eos: 2
        query_value: 3
        query_policy: 4
        query_index: 5
    placeholder:
        node: 6
        edge: 7
    variable:
        node:
            v1: 10
            v2: 11
            v3: 12
            v4: 13
            v5: 14
        edge:
            e1: 15
            e2: 16
            e3: 17
            e4: 18
            e5: 19
    constant:
        '1': 21
        '2': 22
        '3': 23
        '4': 24
        '5': 25
        '(1/2)': 26
        '(1/3)': 27
        '(1/4)': 28
        '(1/5)': 29
    coefficient: 30
    operator:
        binary:
            add: 31
            sub: 32
            mul: 33
            div: 34
            pow: 35 # x^y
            # rac: 36 # x^(1/y)
            regular: 37
        unary:
            neg: 38
            exp: 39
            logabs: 40
            sin: 41
            cos: 42
            tan: 43
            abs: 44
            inv: 45
            sqrtabs: 46
            pow2: 47
            pow3: 48
            # sinh: 49
            # cosh: 50
            tanh: 51
            sigmoid: 52 # 1/(1+exp(-x))
            aggr: 53
            sour: 54
            targ: 55
"""))