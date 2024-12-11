import numbers
import numpy as np
import sympy as sp
from typing import List, Dict, Literal, Union
from functools import reduce
from scipy.optimize import minimize
import logging

logger = logging.getLogger('ND2.symbols')

# ignore RuntimeWarning: invalid value encountered in add/sub/mul/divide
np.seterr(invalid='ignore')

# ignore RuntimeWarning: overflow encountered in exp
np.seterr(over='ignore')

# ignore RuntimeWarning: divide by zero encountered in power
np.seterr(divide='ignore')


class Symbol:
    n_operands = None
    def __init__(self, *operands):
        self.parent = None
        self.child_idx = None

        operands = list(operands) if len(operands) else [Empty() for _ in range(self.n_operands)]
        for idx, op in enumerate(operands):
            if isinstance(op, (float, int, np.ndarray, numbers.Number)):
                operands[idx] = Number(op)

        self.operands = operands
        assert len(self.operands) == self.n_operands

        for idx, operand in enumerate(self.operands):
            operand.parent = self
            operand.child_idx = idx

        self.nettype = 'unknown'
        op_nettypes = set([op.nettype for op in self.operands])
        if op_nettypes - {'unknown', 'scalar', 'node', 'edge'}:
            raise ValueError(f'Unknown nettype in {self.__class__.__name__}')
        elif 'node' in op_nettypes and 'edge' in op_nettypes:
            raise ValueError(f'Inconsistent nettype in {self.__class__.__name__}')
        elif 'node' in op_nettypes:
            self.set_nettype('node')
        elif 'edge' in op_nettypes:
            self.set_nettype('edge')
        elif 'scalar' in op_nettypes and 'unknown' not in op_nettypes:
            self.set_nettype('scalar')

    def __repr__(self):
        return self.to_str()

    def __str__(self):
        return self.to_str()

    def __len__(self):
        return 1 + sum(len(operand) for operand in self.operands)
    
    @classmethod
    def create_instance(cls, *operands):
        return cls(*operands)

    def to_str(self, **kwargs):
        """
        Args:
        - raw:bool=False, whether to return the raw format
        - number_format:str='', can be '0.2f'
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        - latex:bool=False, whether to return the latex format
        - skeleton:bool=False, whether to ignore the concrete values of Number
        """
        name = self.__class__.__name__
        if not kwargs.get('raw'): name = name.lower()
        return f'{name}({", ".join(x.to_str(**kwargs) for x in self.operands)})'

    def to_tree(self, **kwargs):
        """
        Args:
        - number_format:str='', can be '0.2f'
        """
        if self.n_operands == 0: return f'{self.to_str(**kwargs)} ({self.nettype})'
        name = f'{self.__class__.__name__} ({self.nettype})'
        children = [operand.to_tree(**kwargs) for operand in self.operands]
        for idx, child in enumerate(children):
            children[idx] = ('├ ' if idx < len(children)-1 else '└ ') + child.replace('\n', '\n' + ('┆ ' if idx < len(children)-1 else '  '))
        return name + '\n' + '\n'.join(children)

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def preorder(self):
        yield self
        for operand in self.operands:
            yield from operand.preorder()
    
    def postorder(self):
        for operand in self.operands:
            yield from operand.postorder()
        yield self
    
    def __add__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Add(self, other)

    def __radd__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Add(other, self)

    def __sub__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Sub(self, other)

    def __rsub__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Sub(other, self)

    def __mul__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Mul(self, other)

    def __rmul__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Mul(other, self)

    def __truediv__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Div(self, other)
    
    def __rtruediv__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Div(other, self)

    def __pow__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        if other == 2.0: return Pow2(self)
        if other == 3.0: return Pow3(self)
        if other == 0.5: return Sqrt(self)
        if other == -1.0: return Inv(self)
        return Pow(self, other)
    
    def __rpow__(self, other):
        if isinstance(other, (numbers.Number, np.ndarray)): other = Number(other)
        return Pow(other, self)
    
    def __neg__(self):
        return Neg(self)

    def is_constant(self, **kwargs):
        return all([op.is_constant(**kwargs) for op in self.operands])
    
    def replace(self, symbol:'Symbol'):
        if self.parent is None: return symbol
        self.parent.operands[self.child_idx] = symbol
        symbol.parent = self.parent
        symbol.child_idx = self.child_idx
        self.parent = None
        self.child_idx = None
        return symbol
    
    def copy(self):
        copy = self.__class__(*[op.copy() for op in self.operands])
        copy.nettype = self.nettype
        return copy
    
    def fit(self, X:Dict[str,np.ndarray], y:np.ndarray, maxiter=30, method='BFGS'):
        # 对 float32 报警
        float32 = []
        if y.dtype == np.float32: float32.append('y')
        for key, value in X.items():
            if value.dtype == np.float32:
                float32.append(key)
        if len(float32):
            logger.warning(f'{float32} is float32, which may cause numerical instability')

        # 创建一个替身，对这个替身的优化更加容易，fit 它的 fitable Number 即是原来的 fitable Number
        sutando = self.create_sutando(**X)
        if isinstance(sutando, Number) and not sutando.fitable: return self # 没有 fitable Number
        parameters = [op for op in sutando.preorder() if isinstance(op, Number) and op.fitable]
        def set_params(params):
            p = 0
            for param in parameters:
                param.value = params[p:p+param.value.size].reshape(param.value.shape)
                p += param.value.size
        def loss(params):
            set_params(params)
            return np.mean((y - sutando.eval(**X)) ** 2)
        x0 = np.concatenate([param.value.flatten() for param in parameters])
        res = minimize(loss, x0, method=method, options={'maxiter': maxiter})
        set_params(res.x)
        return self
    
    def set_nettype(self, nettype:Literal['node', 'edge', 'scalar']):
        if self.nettype == 'scalar': return
        if self.nettype == 'unknown': self.nettype = nettype
        if self.nettype != nettype:
            raise ValueError(f'Inconsistent nettype in {self.__class__.__name__}')
        for op in self.operands:
            if op.nettype == 'unknown':
                op.set_nettype(self.nettype)

    def create_sutando(self, *args, **kwargs) -> 'Symbol':
        """ 
        使用启发式的方法创建一个替身，与 self 共享 fitable Number
        替身的形式更加简洁，能够更快速地被 fit，且 fit 过程中 self 的 fitable Number 也会被更新
        """
        if self.n_operands == 0:
            if isinstance(self, Number) and self.fitable: return self  # 需要拟合的量
            else: return Number(self.eval(*args, **kwargs), self.nettype, fitable=False)  # 不需要拟合的量
        sutando_operands = [op.create_sutando(*args, **kwargs) for op in self.operands]
        if all(isinstance(op, Number) and not op.fitable for op in sutando_operands):  # 没有 fitable Number 的子公式
            return Number(self.__class__(*sutando_operands).eval(*args, **kwargs), self.nettype, fitable=False)
        return self.__class__(*sutando_operands)  # 有 fitable Number 且难以继续简化的子公式


class Empty(Symbol):
    n_operands = 0
    def to_str(self, **kwargs):
        if kwargs.get('raw', False): return 'Empty()'
        if kwargs.get('latex', False): return r'\square'
        return '?'

    def eval(self, *args, **kwargs):
        raise ValueError('Incomplete Equation Tree')

    def is_constant(self, **kwargs):
        return False


class Number(Symbol):
    n_operands = 0
    def __init__(self, value, nettype:Union[None, Literal['node', 'edge', 'scalar']]=None, fitable=True):
        super().__init__()
        self.value = np.array(value)
        if nettype: self.nettype = nettype
        self.fitable = fitable

    def to_str(self, **kwargs):
        if kwargs.get('raw', False): 
            return f'Number({np.array(self.value).tolist()}, "{self.nettype}", {self.fitable})'
        if kwargs.get('skeleton', False):
            return rf'\square' if kwargs.get('latex') else 'C'
        fmt = kwargs.get('number_format', '')
        if isinstance(self.value, (numbers.Number)) or self.value.size == 1:
            content = f'{self.value:{fmt}}'
        elif kwargs.get('latex', False):
            content = rf'\left<{np.mean(self.value):{fmt}}\right>'
        else:
            content = f'<{np.mean(self.value):{fmt}} (+{np.std(self.value):{fmt}})>'
        return content if self.fitable else f'Constant({content})'
        
    def __eq__(self, value: Union[int, float]) -> bool:
        return self.value == value

    def eval(self, *args, **kwargs):
        return self.value

    def is_constant(self, **kwargs):
        return True
    
    def copy(self):
        from copy import deepcopy
        return self.__class__(deepcopy(self.value), self.nettype, self.fitable)


class Variable(Symbol):
    n_operands = 0
    def __init__(self, name, nettype:Union[None, Literal['node', 'edge', 'scalar']]=None):
        super().__init__()
        self.name = name
        if nettype: self.nettype = nettype
    
    def to_str(self, **kwargs):
        if kwargs.get('raw', False): return f'Variable("{self.name}", "{self.nettype}")'
        if kwargs.get('latex', False): return f'{self.name[0]}_{{{self.name[1:]}}}' if len(self.name) > 1 else self.name
        return self.name 
    
    def eval(self, *args, **kwargs):
        return kwargs[self.name]
        # return eval(self.name, globals(), kwargs)

    def is_constant(self, **kwargs):
        return self.name in kwargs
    
    def copy(self):
        return self.__class__(self.name, self.nettype)


class Add(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        return f'{x1} + {x2}'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) + self.operands[1].eval(*args, **kwargs)
   
    @classmethod
    def create_instance(self, *operands):
        add = [operand for operand in operands if operand.__class__ != Neg]
        sub = [operand.operands[0] for operand in operands if operand.__class__ == Neg]
        if len(sub) == 0: 
            return reduce(lambda x, y: Add(x, y), add)
        elif len(add) == 0:
            return Neg(reduce(lambda x, y: Add(x, y), sub))
        else: 
            return Sub(reduce(lambda x, y: Add(x, y), add), reduce(lambda x, y: Add(x, y), sub))


class Sub(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        if self.operands[1].__class__ in [Add, Sub]:
            x2 = rf'\left({x2}\right)' if kwargs.get('latex', False) else f'({x2})'
        return f'{x1} - {x2}'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) - self.operands[1].eval(*args, **kwargs)


class Mul(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub]: 
            x1 = rf'\left({x1}\right)' if kwargs.get('latex', False) else f'({x1})'
        if self.operands[1].__class__ in [Add, Sub]: 
            x2 = rf'\left({x2}\right)' if kwargs.get('latex', False) else f'({x2})'
        if kwargs.get('omit_mul_sign', False): 
            if self.operands[1].__class__ in [Add, Sub]: return f'{x1}{x2}'
            if isinstance(self.operands[0], Number) and isinstance(self.operands[1], Variable): return f'{x1}{x2}'
                
        return f'{x1} * {x2}' if not kwargs.get('latex', False) else rf'{x1} \times {x2}'
    
    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) * self.operands[1].eval(*args, **kwargs)
   
    @classmethod
    def create_instance(self, *operands):
        if operands[0] == -1: return Neg(Mul.create_instance(*operands[1:]))
        numer = [operand for operand in operands if operand.__class__ != Inv]
        denom = [operand.operands[0] for operand in operands if operand.__class__ == Inv]
        if len(denom) == 0: 
            return reduce(lambda x, y: Mul(x, y), numer)
        elif len(numer) == 0:
            return Inv(reduce(lambda x, y: Mul(x, y), denom))
        else: 
            return Div(reduce(lambda x, y: Mul(x, y), numer), reduce(lambda x, y: Mul(x, y), denom))


class Div(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        if kwargs.get('latex', False): return rf'\frac{{{x1}}}{{{x2}}}'
        if self.operands[0].__class__ in [Add, Sub]:
            x1 = rf'\left({x1}\right)' if kwargs.get('latex', False) else f'({x1})'
        if self.operands[1].__class__ in [Add, Sub, Mul, Div, Inv]:
            x2 = rf'\left({x2}\right)' if kwargs.get('latex', False) else f'({x2})'
        return f'{x1} / {x2}'
    
    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) / self.operands[1].eval(*args, **kwargs)


class Pow(Symbol):
    n_operands = 2
    def to_str(self, **kwargs):
        x1, x2 = self.operands[0].to_str(**kwargs), self.operands[1].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub, Mul, Div, Pow, Neg, Inv, Pow2, Pow3]:
            x1 = rf'\left({x1}\right)' if kwargs.get('latex', False) else f'({x1})'
        if kwargs.get('latex', False): 
            return rf'{x1}^{{{x2}}}'
        if self.operands[1].__class__ in [Add, Sub, Mul, Div, Inv]:
            x2 = rf'\left({x2}\right)' if kwargs.get('latex', False) else f'({x2})'
        return f'{x1} ** {x2}'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) ** self.operands[1].eval(*args, **kwargs)
    
    @classmethod
    def create_instance(self, *operands):
        if operands[1] == 0.5: return Sqrt(operands[0])
        if operands[1] == -1: return Inv(operands[0])
        if operands[1] == 2: return Pow2(operands[0])
        if operands[1] == 3: return Pow3(operands[0])
        return Pow(*operands)


class Cat(Symbol):
    n_operands = 2
    def __init__(self, *operands):
        super().__init__(*operands)
        assert all([isinstance(operand, Number) for operand in operands])

    def __str__(self):
        return f'{self.operands[0]}{self.operands[1]}'

    def eval(self, *args, **kwargs):
        return int(str(self))


class Max(Symbol):
    n_operands = 2
    def eval(self, *args, **kwargs):
        x1 = self.operands[0].eval(*args, **kwargs)
        x2 = self.operands[1].eval(*args, **kwargs)
        return np.maximum(x1, x2)

    def create_instance(self, *operands):
        return reduce(lambda x, y: Max(x, y), operands)


class Min(Symbol):
    n_operands = 2
    def eval(self, *args, **kwargs):
        x1 = self.operands[0].eval(*args, **kwargs)
        x2 = self.operands[1].eval(*args, **kwargs)
        return np.minimum(x1, x2)
    
    def create_instance(self, *operands):
        return reduce(lambda x, y: Min(x, y), operands)


class Sin(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.sin(self.operands[0].eval(*args, **kwargs))
sin = lambda x: Sin(x)


class Cos(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.cos(self.operands[0].eval(*args, **kwargs))
cos = lambda x: Cos(x)


class Tan(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.tan(self.operands[0].eval(*args, **kwargs))
tan = lambda x: Tan(x)


class Log(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.log(self.operands[0].eval(*args, **kwargs))
log = lambda x: Log(x)


class Exp(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.exp(self.operands[0].eval(*args, **kwargs))
exp = lambda x: Exp(x)


class Arcsin(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.arcsin(self.operands[0].eval(*args, **kwargs))
arcsin = lambda x: Arcsin(x)


class Arccos(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.arccos(self.operands[0].eval(*args, **kwargs))
arccos = lambda x: Arccos(x)


class Arctan(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.arctan(self.operands[0].eval(*args, **kwargs))
arctan = lambda x: Arctan(x)


class Sqrt(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.sqrt(self.operands[0].eval(*args, **kwargs))    
sqrt = lambda x: Sqrt(x)


class Abs(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return abs(self.operands[0].eval(*args, **kwargs))

class Neg(Symbol):
    n_operands = 1
    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub]: x = f'({x})'
        return f'-{x}'
    
    def eval(self, *args, **kwargs):
        return -self.operands[0].eval(*args, **kwargs)

class Inv(Symbol):
    n_operands = 1
    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if kwargs.get('latex', False): return rf'\frac{{1}}{{{x}}}'
        if self.operands[0].__class__ in [Add, Sub, Mul, Div]: x = f'({x})'
        return f'1 / {x}'

    def eval(self, *args, **kwargs):
        return 1/self.operands[0].eval(*args, **kwargs)

class Pow2(Symbol):
    n_operands = 1
    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub, Mul, Div, Pow, Neg, Inv, Pow2, Pow3]: x = f'({x})'
        return f'{x} ** 2' if not kwargs.get('latex', False) else f'{x}^2'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) ** 2

class Pow3(Symbol):
    n_operands = 1
    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if self.operands[0].__class__ in [Add, Sub, Mul, Div, Pow, Neg, Inv, Pow2, Pow3]: x = f'({x})'
        return f'{x} ** 3' if not kwargs.get('latex', False) else f'{x}^3'

    def eval(self, *args, **kwargs):
        return self.operands[0].eval(*args, **kwargs) ** 3

class Tanh(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return np.tanh(self.operands[0].eval(*args, **kwargs))

class Sigmoid(Symbol):
    n_operands = 1
    def eval(self, *args, **kwargs):
        return 1 / (1 + np.exp(-self.operands[0].eval(*args, **kwargs)))

class Regular(Symbol):
    n_operands = 2
    def eval(self, *args, **kwargs):
        x1 = self.operands[0].eval(*args, **kwargs)
        x2 = self.operands[1].eval(*args, **kwargs)
        return x1 ** x2 / (1 + x1 ** x2)

class Sour(Symbol):
    n_operands = 1
    def __init__(self, *operands):
        super().__init__(*operands)
        self.nettype = 'edge'
        self.operands[0].set_nettype('node')

    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if kwargs.get('latex', False): return rf'\phi_s({x})'
        return f'{self.__class__.__name__}({x})'

    def eval(self, *args, **kwargs):
        """(*, n_nodes) -> (*, n_edges)"""
        x = self.operands[0].eval(*args, **kwargs)
        A, G = kwargs['A'], kwargs['G'] # (V, V), (E, 2)
        V, E = A.shape[0], G.shape[0]
        if isinstance(x, numbers.Number) or x.size == 1:
            x = np.full((V,), x)
        elif self.operands[0].nettype == 'scalar': 
            if x.shape[-1] != 1: x = x[..., np.newaxis]
            x = np.repeat(x, V, axis=-1)
        return x[..., G[:, 0]] # (*, V) -> (*, E)


class Targ(Symbol):
    n_operands = 1
    def __init__(self, *operands):
        super().__init__(*operands)
        self.nettype = 'edge'
        self.operands[0].set_nettype('node')

    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if kwargs.get('latex', False): return rf'\phi_t({x})'
        return f'{self.__class__.__name__}({x})'

    def eval(self, *args, **kwargs):
        """(*, n_nodes) -> (*, n_edges)"""
        x = self.operands[0].eval(*args, **kwargs)
        A, G = kwargs['A'], kwargs['G'] # (V, V), (E, 2)
        V, E = A.shape[0], G.shape[0]
        if isinstance(x, numbers.Number) or x.size == 1:
            x = np.full((V,), x)
        elif self.operands[0].nettype == 'scalar': 
            if x.shape[-1] != 1: x = x[..., np.newaxis]
            x = np.repeat(x, V, axis=-1)
        return x[..., G[:, 1]] # (*, V) -> (*, E)


class Aggr(Symbol):
    n_operands = 1
    def __init__(self, *operands):
        super().__init__(*operands)
        self.nettype = 'node'
        self.operands[0].set_nettype('edge')

    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if kwargs.get('latex', False): return rf'\rho({x})'
        return f'{self.__class__.__name__}({x})'

    def eval(self, *args, **kwargs):
        """(*, n_edges) -> (*, n_nodes)"""
        x = self.operands[0].eval(*args, **kwargs)
        A, G = kwargs['A'], kwargs['G'] # (V, V), (E, 2)
        V, E = A.shape[0], G.shape[0]
        if isinstance(x, numbers.Number) or x.size == 1:
            x = np.full((E,), x)
        elif self.operands[0].nettype == 'scalar': 
            if x.shape[-1] != 1: x = x[..., np.newaxis]
            x = np.repeat(x, E, axis=-1)
        y = np.zeros((*x.shape[:-1], V)) # (*, V)
        for edge_idx in range(E):
            y[..., G[edge_idx, 1]] += x[..., edge_idx]
        return y


class Rgga(Symbol):
    n_operands = 1
    def __init__(self, *operands):
        super().__init__(*operands)
        self.nettype = 'node'
        self.operands[0].set_nettype('edge')

    def to_str(self, **kwargs):
        x = self.operands[0].to_str(**kwargs)
        if kwargs.get('latex', False): return rf'\rho^{-1}({x})'
        return f'{self.__class__.__name__}({x})'

    def eval(self, *args, **kwargs):
        """(*, n_edges) -> (*, n_nodes)"""
        x = self.operands[0].eval(*args, **kwargs)
        A, G = kwargs['A'], kwargs['G'] # (V, V), (E, 2)
        V, E = A.shape[0], G.shape[0]
        if isinstance(x, numbers.Number) or x.size == 1:
            x = np.repeat(x[np.newaxis], E, axis=-1) # (*, E)
        elif self.operands[0].nettype == 'scalar': 
            if x.shape[-1] != 1: x = x[..., np.newaxis]
            x = np.repeat(x, E, axis=-1)
        y = np.zeros((*x.shape[:-1], V)) # (*, V)
        for edge_idx in range(E):
            y[..., G[edge_idx, 0]] += x[..., edge_idx]
        return y


class Readout(Symbol):
    n_operands = 1
    def __init__(self, *operands):
        super().__init__(*operands)
        self.nettype = 'scalar'
    
    def eval(self, *args, **kwargs):
        x = self.operands[0].eval(*args, **kwargs)
        return np.sum(x, axis=-1, keepdims=True)
