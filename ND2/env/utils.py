import numpy as np
import sympy as sp
from typing import List
from .symbols import *


def prefix2eqtree(prefix:List[str]):
    def foo(idx):
        item, idx = eval(prefix[idx]), idx+1
        operands = []
        if type(item) == int: 
            operands, item = [item], Number
    
        for _ in range(item.n_operands):
            operand, idx = foo(idx)
            operands.append(operand)
        return item(*operands), idx

    eqtree, idx = foo(0)
    assert idx == len(prefix)
    return eqtree


_sympy2symbol = {
    'Add': Add,
    'Mul': Mul,
    'Pow': Pow,
    'Abs': Abs,
    'sin': Sin,
    'cos': Cos,
    'exp': Exp,
    'log': Log,
    'aggr': Aggr,
    'rgga': Rgga,
    'sour': Sour,
    'targ': Targ,
}
def sympy2eqtree(expr:sp.Expr):
    if expr.is_Atom:
        if expr.is_Symbol:
            return Variable(expr.name)
        elif expr.is_infinite:
            return Number(float('inf'))
        else:
            return Number(float(expr))
    symbol = _sympy2symbol[expr.func.__name__]
    operands = [sympy2eqtree(arg) for arg in expr.args]
    # if all(isinstance(op, Number) for op in operands):
    #     return Number(symbol.create_instance(*operands).eval())
    return symbol.create_instance(*operands)


def check_nettype(eqtree:Symbol, root:Literal['node', 'edge']='node'):
    if isinstance(eqtree, Variable): return eqtree.nettype == root
    if isinstance(eqtree, Number): return True
    if eqtree.__class__ in [Aggr, Rgga]: return eqtree.nettype == root == 'node' and check_nettype(eqtree.operands[0], 'edge')
    if eqtree.__class__ in [Sour, Targ]: return eqtree.nettype == root == 'edge' and check_nettype(eqtree.operands[0], 'node')
    return all(check_nettype(op, root) for op in eqtree.operands)


def simplify(eqtree:Symbol):
    try:
        variables = set([v.name for v in eqtree.preorder() if isinstance(v, Variable)])
        local_dict = {v:sp.Symbol(v) for v in variables}
        eq = str(eqtree)
        eq = sp.parse_expr(eq, local_dict=local_dict)
        eq = sp.simplify(eq)
        eq = sympy2eqtree(eq)
        return eq
    except Exception as e:
        logger.warning(f'Failed to simplify {eqtree}: {e}')
        return eqtree
