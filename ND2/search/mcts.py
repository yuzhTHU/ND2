import os
import sys
import time
import torch
import logging
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
from tqdm import tqdm
from typing import List, Dict, Tuple, Literal
from copy import deepcopy
from functools import partial
from itertools import compress
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ..model import NDformer
from ..dataset import Tokenizer
from ..GDExpr import GDExprClass, GDExpr
from .reward_solver import RewardSolver
from ..utils import NamedTimer, Timer, AbsTimer
logger = logging.getLogger('ND2.MCTS')

if '--test_wo_redundant_op' in sys.argv:
    GDExpr = GDExprClass('./config/basic_wo_redundant_op.yaml')

if '--test_测试韬哥说的低维符号回归任务' in sys.argv:
    GDExpr = GDExprClass('./config/basic_low.yaml')

if '--no_triangle' in sys.argv:
    GDExpr = GDExprClass('./config/basic_wo_triangle_op.yaml')

DEBUG_SPLIT_FLAG = '--split' in sys.argv
DEBUG_NORESOLVE_FLAG = '--noresolve' in sys.argv

class MCTS(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, 
                 rewarder:RewardSolver=None, 
                 ndformer:NDformer=None,
                 vars_node:List[str]=[],
                 vars_edge:List[str]=[],
                 binary:List[str] = ['add', 'sub', 'mul', 'div', 'pow', 'regular'],
                 unary:List[str] = ['neg', 'abs', 'inv', 
                                    'exp', 'logabs', 
                                    'sin', 'cos', 'tan', 
                                    'sqrtabs', 'pow2', 'pow3', 
                                    'tanh', 'sigmoid',
                                    'aggr', 'sour', 'targ'],
                 constant:List[str] = ['1', '2', '3', '4', '5', 
                                       '(1/2)', '(1/3)', '(1/4)', '(1/5)'],
                 log_per_episode:int=100,
                 log_per_second:int=10,
                 select_tempreture:float=0.0,
                 beam_size:int=20,
                 lambda_rollout:float=0.5,
                 max_token_num:int=30,
                 max_coeff_num:int=2,
                 random_state:int=0,
                 c_puct:float=1.41,
                 repeat_times:int=10,
                 use_random_simulate:bool=False,
                 **kwargs
                 ): 
        """
        Arguments:
        - Xv: Dict[str, np.ndarray], for each X['variable'], whose shape = (Time, Node)
        - Xe: Dict[str, np.ndarray], for each X['variable'], whose shape = (Time, Edge)
        - Y: np.ndarray, shape = (Time, Node or Edge)
        - G: np.ndarray, shape = (Edge, 2)
        - A: np.ndarray, shape = (Node, Node)
        - root_type: str, 'node' (if Y.shape == (Time, Node)) or 'edge' (if Y.shape == (Time, Edge))
        - rewarder: RewardSolver
        - ndformer: NDformer
        """
        super().__init__()
        self.rewarder = rewarder or RewardSolver()
        self.ndformer = ndformer
        self.vars_node = vars_node
        self.vars_edge = vars_edge
        self.binary = binary
        self.unary = unary
        self.constant = constant
        self.log_per_episode = log_per_episode
        self.log_per_second = log_per_second
        self.tempreture = select_tempreture
        self.beam_size = beam_size
        self.lambda_rollout = lambda_rollout
        self.max_token_num = max_token_num
        self.max_coeff_num = max_coeff_num
        self.random_state = random_state or np.random.randint(1, 35167)
        self.c_puct = c_puct
        self.repeat_times = repeat_times
        self.use_random_simulate = use_random_simulate
        
        self.tabula_rasa = (ndformer is None)
        self.actions = vars_node + vars_edge + binary + unary + constant
        self.action2idx = {action: idx for idx, action in enumerate(self.actions)}
        self.n_actions = len(self.actions)

        if kwargs: logger.warning(f'Unused arguments: {kwargs} in MCTS')
        assert select_tempreture >= 0
        assert beam_size >= 1
        assert 0 <= lambda_rollout <= 1
        assert len(set(self.actions)) == len(self.actions)

        # self.action2token = {var:f'v{idx+1}' for idx, var in enumerate(vars_node)} + \
        #                     {var:f'e{idx+1}' for idx, var in enumerate(vars_edge)} + \
        #                     {op:op for op in self.binary + self.unary + self.constant}
        # self.token2action = {v: k for k, v in self.action2token.items()}
        # self.idx2action = {v: k for k, v in self.action2idx.items()}
        # if self.ndformer is not None:
        #     self.ndformer.eval()
        #     with torch.no_grad():
        #         self.data_emb = self.ndformer.encode(self.root_type, self.var_dict, sample=True)
        
        self.rewards = dict() # tuple(terminal prefix) -> reward
        self.MC_Tree = dict() # tuple(prefix) -> tuple(Q, N, P, V)
                              # Q: max possible reward of each child
                              # N: visit count of each child
                              # P: prior probability of each child
                              # V: prior value (or reward value if terminal) of current state
        self.best_result = defaultdict(lambda: np.nan) | {'reward': -np.inf}
        self.search_history = [] # List of tuple([current rewards], current best reward, current best prefix)

        self.named_timer = NamedTimer()
        self.episode_timer = Timer()
        self.state_timer = Timer()
        self.eq_timer = AbsTimer()

        # if self.config.get('ObsSNR', None):
        #     noise = 10 ** (-self.config.ObsSNR / 20) * self.var_dict['out'].std()
        #     self.var_dict['out_raw'] = deepcopy(self.var_dict['out'])
        #     self.var_dict['out'] += np.random.normal(0, noise, self.var_dict['out'].shape)
        #     logger.info(f'Add Noise with an SNR of {self.config.ObsSNR:.1f}dB ({10 ** (-self.config.ObsSNR / 20):.2%}) to Data')
        
        # if self.config.get('spurious_link_ratio', None):
        #     self.var_dict['A_raw'] = deepcopy(self.var_dict['A'])
        #     self.var_dict['G_raw'] = deepcopy(self.var_dict['G'])
        #     A = self.var_dict['A']
        #     non_edge_list = np.stack(np.nonzero(A==0), axis=-1)
        #     E = np.clip(int(self.config.spurious_link_ratio * A.sum()), 0, len(non_edge_list))
        #     G_ = non_edge_list[np.random.choice(np.arange(len(non_edge_list)), E, replace=False)]
        #     A[G_[:, 0], G_[:, 1]] = 1
        #     self.var_dict['A'] = A
        #     self.var_dict['G'] = np.stack(np.nonzero(A), axis=-1)
        #     logger.info(f'Add Spurious Link with a Ratio {self.config.spurious_link_ratio:.1%} to Data (E={self.var_dict["A_raw"].sum()}->{self.var_dict["A"].sum()})')

        # if self.config.get('missing_link_ratio', None):
        #     self.var_dict['A_raw'] = deepcopy(self.var_dict['A'])
        #     self.var_dict['G_raw'] = deepcopy(self.var_dict['G'])
        #     G = self.var_dict['G']
        #     E = np.clip(int((1-self.config.missing_link_ratio) * len(G)), 0, len(G))
        #     G_ = G[np.random.choice(np.arange(len(G)), E, replace=False)]
        #     A = np.zeros_like(self.var_dict['A'])
        #     A[G_[:, 0], G_[:, 1]] = 1
        #     self.var_dict['A'] = A
        #     self.var_dict['G'] = G_
        #     logger.info(f'Add Missing Link with a Ratio {self.config.missing_link_ratio:.1%} to Data (E={self.var_dict["A_raw"].sum()}->{self.var_dict["A"].sum()})')

        # if self.config.get('diff_as_out', None): 
        #     self.var_dict['out_raw'] = deepcopy(var_dict['out'])
        #     dx = np.diff(self.var_dict[config.diff_as_out], axis=0)
        #     if config.diff_mod_pi:
        #         dx = (np.mod(dx + np.pi, np.pi*2) - np.pi)
        #     self.var_dict['out'] = dx / 0.01
        #     for key in set(self.var_dict.keys()) - {'A', 'G', 'out'}:
        #         self.var_dict[key] = self.var_dict[key][:-1]


    def fit(self, 
            root_prefix:List[str], 
            episode_limit:int=1_000_000,
            time_limit:int=None,
            early_stop=lambda best_result: best_result['ACC4'] > 0.99,
            ):
        """
        Arguments:
        - root_prefix: e.g.1: ['node'], e.g.2: ['add', 'node', 'aggr', 'sour', 'node']
        """
        # self.var_dict = {**Xv, **Xe, 'out': Y, 'A': A, 'G': G}
        # self.var_list = list(Xv.keys()) + list(Xe.keys())
        # if '--no_e1' in sys.argv: self.var_list = list(set(self.var_list) - {'e1'})

        self.start_time = time.time()
        self.named_timer.reset()
        self.episode_timer.reset()
        self.state_timer.reset()
        self.eq_timer.reset()
        
        self.expand([root_prefix])
        for episode in range(1, episode_limit+1):
            self.named_timer.add('pre')

            # Select
            routes = self.select(root_prefix)
            exists_flag = np.array([tuple(route[-1][1]) in self.MC_Tree for route in routes])
            states_to_expand = [route[-1][1] for route, e in zip(routes, exists_flag) if not e]
            self.named_timer.add('select')
            
            # Expand
            self.expand(states_to_expand)
            self.named_timer.add('expand')
            
            # Simulate
            Q = np.full((len(routes),), np.nan)
            Q[~exists_flag] = self.simulate(states_to_expand)
            if np.any(exists_flag):
                Q[exists_flag] = self.get_rewards([route[-1][1] for route, e in zip(routes, exists_flag) if e])
            if not np.isfinite(Q).all(): 
                logger.error(f'Invalid Value in Q: {Q}')
                Q[~np.isfinite(Q)] = 0.0
            self.named_timer.add('simulate')

            # Backpropagate
            self.backpropagate(routes, Q)
            self.named_timer.add('backpropagate')

            self.episode_timer.add(1)
            self.state_timer.add(len(states_to_expand))
            self.eq_timer.set(len(self.rewards))
            # self.search_history.append((Q.tolist(), self.best_result['reward'], self.best_result['prefix']))
            if self.log_per_episode and (episode % self.log_per_episode == 0) or \
               self.log_per_second and (self.episode_timer.time > self.log_per_second):
                log = {
                    'Episode': f'{episode}',
                    'Speed': f'{self.episode_timer.pop():.2f} episode/s, {self.state_timer.pop():.2f} expanded node/s, {self.eq_timer.pop():.2f} eq/s',
                    'Time Usage': self.named_timer.pop(),
                    # 'Current Best': f'{self.best_result["reward"]:.2f}',
                    # 'Reward': f'{self.best_result["reward"]:.2f}',
                    # 'R2': f'{self.best_result["R2"]:.4f}',
                    # 'RMSE': f'{self.best_result["RMSE"]:.3e}',
                    # 'MAPE': f'{self.best_result["MAPE"]:.2%}',
                    # 'wMAPE': f'{self.best_result["wMAPE"]:.2%}',
                    # 'sMAPE': f'{self.best_result["sMAPE"]:.2%}',
                    # 'Acc2': f'{self.best_result["ACC2"]:.2%}',
                    # 'Acc3': f'{self.best_result["ACC3"]:.2%}',
                    # 'Acc4': f'{self.best_result["ACC4"]:.2%}',
                    # 'Complexity': f'{self.best_result["complexity"]}',
                    **self.best_result,
                    'Current': GDExpr.prefix2str(states_to_expand[0]) if states_to_expand else 'None',
                }
                logger.info(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))

            if early_stop is not None and early_stop(self.best_result):
                logger.note(f'Early stop at episode {episode}.')
                break
            
            if time_limit is not None and time.time() - self.start_time > time_limit:
                logger.note(f'Time limit reached at episode {episode}.')
                break

    def select(self, root_prefix:List[str]) -> List[List[tuple]]:
        if self.beam_size == 1:
            return [self.plain_select(root_prefix)]
        else:
            return self.beam_select(root_prefix)
    
    def plain_select(self, root_prefix:List[str]) -> List[tuple]:
        """
        Returns:
        - route: List[tuple], a list of (action, state)
            state0=root_prefix -> action1 -> state1 -> ... -> actionN -> stateN (leaf or terminal)
            (action0 is None, state0 is root_prefix)
        """
        route = [(None, root_prefix)]
        state = route[-1][1]
        while tuple(state) in self.MC_Tree:
            mask = self.get_mask(state)
            if not mask.any(): break # terminal node
            M = np.where(mask, 0., -np.inf)
            UCT = self.get_UCT(state) + M
            action = self.actions[np.argmax(UCT)]
            state = self.act(state, action)
            route.append((action, state))
        return route

    def beam_select(self, root_prefix:List[str]) -> List[List[tuple]]:
        """
        Returns:
        - routes: List[List[tuple]] (len=1)
            List of routes, each route is a list of (action, state)
            A route: state0=root_prefix -> action1 -> state1 -> ... -> actionN -> stateN (leaf or terminal)
            (action0 is None, state0 is root_prefix)
        """
        beam_search = [ ([(None, root_prefix)], [], False) ] # route, UCT_route, Is_leaf_or_terminal
        while any(not done for _, _, done in beam_search):
            UCT_routes = []
            # calculate UCT for each route
            for route, UCT_route, done in beam_search:
                if done:
                    UCT_routes.append(UCT_route) # shape: [(1,), (1,), ..., (1,)]
                else:
                    state = route[-1][1]
                    mask = self.get_mask(state)
                    M = np.where(mask, 0., -np.inf)
                    UCT = self.get_UCT(state) + M
                    UCT_routes.append([*UCT_route, UCT]) # shape: [(1,), (1,), ..., (1,), (N,)]
            # find k expanded routes with best mean UCT
            metric = np.concatenate([sum(u) / len(u) for u in UCT_routes], axis=0)
            k = min(self.beam_size, np.isfinite(metric).sum())
            topk_pos = np.argpartition(metric, -k)[-k:]
            split_pos = np.cumsum([0] + [u[-1].shape[0] for u in UCT_routes])
            state_id = np.searchsorted(split_pos, topk_pos, side='right') - 1
            action_id = topk_pos - split_pos[state_id]
            _beam_search = []
            for idx, action in zip(state_id, action_id):
                if UCT_routes[idx][-1].shape[0] == 1: 
                    _beam_search.append(beam_search[idx])
                else:
                    route = beam_search[idx][0]
                    state = self.act(route[-1][1], self.actions[action])
                    done = (tuple(state) not in self.MC_Tree) or (not self.get_mask(state).any())
                    UCT = UCT_routes[idx][-1][action:action+1] # size: (1,)
                    _beam_search.append(([*route, (self.actions[action], state)], [*UCT_routes[idx][:-1], UCT], done))
            beam_search = _beam_search
        routes = [route for route, _, _ in beam_search]
        return routes

    def expand(self, states_to_expand:List[List[str]]):
        """
        Arguments:
        - states_to_expand: List[List[str]], length = M
            M states to expand
        """
        if not self.tabula_rasa: policies = self.get_policy(states_to_expand)[0]
        else:                    policies = [None for _ in states_to_expand]
        for state, policy in zip(states_to_expand, policies):
            assert tuple(state) not in self.MC_Tree
            self.MC_Tree[tuple(state)] = [np.zeros(self.n_actions), np.zeros(self.n_actions), policy]

    def simulate(self, states_to_expand:List[List[str]]) -> np.ndarray:
        """
        Arguments:
        - states_to_expand: List[List[str]], length = M
            M states to expand
        
        Returns:
        - Q: np.ndarray(M)
            Q value of each state
        """
        if self.tabula_rasa or self.use_random_simulate:
            return self.random_simulate(states_to_expand)
        else:
            return self.NN_simulate(states_to_expand)

    def random_simulate(self, states_to_expand:List[List[str]]):
        states = states_to_expand * self.repeat_times
        for idx, state in enumerate(states):
            while True:
                mask = self.get_mask(state)
                if not mask.any(): break
                action = self.actions[np.random.choice(np.nonzero(mask)[0])]
                state = self.act(state, action)
            states[idx] = state
        all_Q = self.get_rewards(states).reshape(self.repeat_times, len(states_to_expand))
        if not np.isfinite(all_Q).all(): 
            logger.error(f'Invalid Value in all_Q: {all_Q}')
            all_Q[~np.isfinite(all_Q)] = -np.inf
        Q = np.max(all_Q, axis=0)
        return Q

    def NN_simulate(self, states_to_expand:List[List[str]]):
        states = states_to_expand * self.repeat_times
        while True:
            self.named_timer.add('simulate.pre')
            policies, terminals = self.get_policy(states)
            self.named_timer.add('simulate.get_policy')
            if not any(terminals): break
            for idx, (state, policy, terminal) in enumerate(zip(states, policies, terminals)):
                if not terminal: continue
                action = self.actions[np.random.choice(self.n_actions, p=policy)]
                states[idx] = self.act(state, action)
            self.named_timer.add('simulate.rollout')
        all_Q = self.get_rewards(states).reshape(self.repeat_times, len(states_to_expand))
        self.named_timer.add('simulate.reward')
        if not np.isfinite(all_Q).all(): 
            logger.error(f'Invalid Value in all_Q: {all_Q}')
            all_Q[~np.isfinite(all_Q)] = -np.inf
        Q = np.max(all_Q, axis=0)
        return Q
    
    def backpropagate(self, routes, Q):
        """
        Arguments:
        - routes: List[List[tuple]] (len=1)
            List of routes, each route is a list of (action, state)
            A route: state0=root_prefix -> action1 -> state1 -> ... -> actionN -> stateN (leaf or terminal)
            (action0 is None, state0 is root_prefix)
        - Q: np.ndarray(len(routes))
            Q value of each route's leaf node
        """
        for q, route in zip(Q, routes):
            cur_state = tuple(route[0][1])
            for action, next_state in route[1:]:
                action_idx = self.action2idx[action]
                self.MC_Tree[cur_state][0][action_idx] = max(q, self.MC_Tree[cur_state][0][action_idx])
                self.MC_Tree[cur_state][1][action_idx] += 1
                cur_state = tuple(next_state)

    def act(self, prefix:List[str], action:str):
        assert action in self.actions
        for idx, type in enumerate(prefix):
            if type in ['node', 'edge']: break
        else: 
            raise ValueError(f'No placeholder in {prefix}')
        
        if action in ['aggr', 'rgga']:
            assert type == 'node', f'{type} != node'
            return [*prefix[:idx], action, 'edge', *prefix[idx+1:]]
        elif action in ['sour', 'targ']:
            assert type == 'edge', f'{type} != edge'
            return [*prefix[:idx], action, 'node', *prefix[idx+1:]]
        elif action in self.binary:
            return [*prefix[:idx], action, type, type, *prefix[idx+1:]]
        elif action in self.unary:
            return [*prefix[:idx], action, type, *prefix[idx+1:]]
        else: # variables / constants / coefficient
            return [*prefix[:idx], action, *prefix[idx+1:]]

    def get_mask(self, prefix:List[str]):
        token_num = len(prefix)
        for idx, type in enumerate(prefix):
            if type in ['node', 'edge']: break
        else:
            return np.zeros((self.n_actions,), dtype=bool)

        mask = np.zeros((self.n_actions,), dtype=bool)
        vectorize = np.vectorize(self.action2idx.get, otypes=[int])
        # Binary operators
        if self.max_token_num - token_num >= 2: mask[vectorize(self.binary)] = 1
        # Unary operators
        if self.max_token_num - token_num >= 1: mask[vectorize(self.unary)] = 1
        if type == 'edge': 
            if 'aggr' in self.actions: mask[vectorize('aggr')] = 0
            if 'rgga' in self.actions: mask[vectorize('rgga')] = 0
        if type == 'node':
            if 'sour' in self.actions: mask[vectorize('sour')] = 0
            if 'targ' in self.actions: mask[vectorize('targ')] = 0
        if self.max_token_num - token_num >= 0:
            # Variables
            if type == 'node': mask[vectorize(self.vars_node)] = 1
            if type == 'edge': mask[vectorize(self.vars_edge)] = 1
            # Constants & Coefficients
            constant_ok = False
            cnt, pos = 1, idx - 1
            while cnt > 0 and pos >= 0 and not constant_ok:
                if prefix[pos] in self.binary: cnt -= 2
                elif prefix[pos] in self.unary: cnt -= 1
                elif prefix[pos] in ['node', 'edge'] + self.vars_node + self.vars_edge: cnt += 1; constant_ok = True
                else: cnt += 1
                pos -= 1
            if constant_ok or (cnt == -1):
                mask[vectorize(self.constant)] = 1
                # if prefix.count(GDExpr.coeff_token) < self.max_coeff_num:
                #     mask[vectorize(GDExpr.coeff_token)] = 1
        return mask

    def get_UCT(self, state:List[str]) -> np.ndarray:
        Q, N, P = self.MC_Tree[tuple(state)]
        if self.tabula_rasa:
            UCT = Q + self.c_puct * np.sqrt(np.log(N.sum().clip(1)) / N.clip(1))
        else:
            UCT = Q + self.c_puct * np.sqrt(N.sum()) * P / (1 + N)
        return UCT

    def get_policy(self, states:List[List[str]]):
        """
        Arguments:
        - states: List of List[str], length = M

        Returns:
        - policies: np.ndarray(M, n_actions), have been softmaxed
        - flags: np.ndarray(M), 1 if forwarded, 0 if non-terminal
        """
        self.named_timer.add('drop')
        if self.tabula_rasa: raise ValueError('Cannot get policies in tabula rasa mode')
        masks = [self.get_mask(state) for state in states]
        policies = [None for _ in states]
        terminals = [m.any() for m in masks]
        if not any(terminals): return policies, terminals
        self.named_timer.add('get_policy.pre')
        tmp = self.ndformer.get_policy(list(compress(states, terminals)), 
                                       self.actions, 
                                       list(compress(masks, terminals)))
        self.named_timer.add('get_policy.forward')
        for idx, policy in zip(np.where(terminals)[0], tmp):
            policies[idx] = policy
        return policies, terminals
        # if getattr(self, 'turn_Cv_and_Ce_to_C', False):
        #     states = deepcopy(states)
        #     for p_idx, p in enumerate(states):
        #         while 'Cv' in p: p[p.index('Cv')] = '<C>'
        #         while 'Ce' in p: p[p.index('Ce')] = '<C>'
        #         states[p_idx] = p
    
    def get_rewards(self, states):
        # if (self.max_workers or 0) == 0:
        rewards = np.array([self.get_reward(state)[0] for state in states])
        # elif self.config.parallelize == 'thread':
        #     with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
        #         results = executor.map(self.get_reward, states)
        #     rewards = np.array(list(results))
        # elif self.config.parallelize == 'process':
        #     raise NotImplementedError
        #     with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
        #         results = executor.map(self.get_reward, states)
        #     rewards, best_results = zip(*results)
        #     rewards = np.array(rewards)
        #     for state, reward, best_result in zip(states, rewards, best_results):
        #         self.rewards[tuple(state)] = reward
        #         if best_result is not None and best_result['reward'] > self.best_result['reward']:
        #             self.best_result = best_result
        # else:
        #     raise ValueError(f'Invalid parallelize method: {self.config.parallelize}')
        if any(rewards >= self.best_result['reward']):
            for idx, state in enumerate(states):
                if rewards[idx] <= self.best_result['reward']: continue
                # Recheck the reward
                rewards[idx], coef_dict = self.get_reward(state, use_cache=False, update_cache=True, sample=False)
                if coef_dict is None: continue
                if rewards[idx] <= self.best_result['reward']: continue
                # Update the best result
                self.best_result = self.rewarder.evaluate(state, coef_dict) | dict(reward=rewards[idx], prefix=state)
                log = {
                    'Update': GDExpr.prefix2str(state),
                    **self.best_result
                }
                logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))

        return rewards

    def get_reward(self, state:List[str], use_cache=True, update_cache=True, sample=True):
        # Too complex
        if len(state) > self.max_token_num: return 0.0, None
        # Non-terminal
        if 'node' in state or 'edge' in state: return 0.0, None
        # Too many coefficients
        if GDExpr.count_coef(state) > self.max_coeff_num: return 0.0, None
        # Cache
        if use_cache and tuple(state) in self.rewards: return self.rewards[tuple(state)], None

        # if getattr(self, 'turn_aggr_to_aggr_Ce', False) and 'aggr' in state:
        #     idx = state.index('aggr')
        #     state = state[:idx] + ['aggr', 'mul', 'Ce'] + state[idx+1:]
        
        # if getattr(self, 'turn_aggr_to_aggr_e1', False):
        #     if 'e1' in state: return 0.0
        #     if 'aggr' in state:
        #         idx = state.index('aggr')
        #         state = state[:idx] + ['aggr', 'mul', 'e1'] + state[idx+1:]

        # if getattr(self, 'force_aggr', False) and 'aggr' not in state: return 0.0
        # if getattr(self, 'turn_C_to_Cv_and_Ce', False):
        #     phd_type = GDExpr.analysis_type(state, 'node')
        #     state = deepcopy(state)
        #     while '<C>' in state: 
        #         idx = state.index('<C>')
        #         state[idx] = 'Cv' if phd_type[idx] == 'node' else 'Ce'
        
        reward, coef_dict = self.rewarder.solve(state, sample=sample)
        # if reward > self.best_result['reward'] and self.rewarder.sample_num is not None:
        #     reward, coef_dict = self.rewarder.solve(state, sample=False, x0=coef_dict)

        if update_cache: self.rewards[tuple(state)] = reward

        return reward, coef_dict
        # if reward > self.best_result['reward']:
        #     # var_dict = deepcopy(self.var_dict)
        #     # if self.config.get('ObsSNR', None) or self.config.get('diff_as_out', None):
        #     #     var_dict['out'] = self.var_dict['out_raw']
        #     # if self.config.get('spurious_link_ratio', None) or self.config.get('missing_link_ratio', None):
        #     #     var_dict['A'] = self.var_dict['A_raw']
        #     #     var_dict['G'] = self.var_dict['G_raw']
        #     result = self.rewarder.evaluate(state, coef_dict)
        #     result['reward'] = reward
        #     result['time'] = time.time() - self.start_time
        #     # result['prefix'] = [coef_list.pop(0) if item == GDExpr.coeff_token else item for item in state]
        #     # result['equation'] = GDExpr.prefix2str(result["prefix"]) if result["prefix"] is not None else "None"
        #     logger.note(f'Update best reward to {reward:.2f} '
        #         # f'[{result["equation"]}] ('
        #         f'R2={result["R2"]:.4f}, '
        #         f'RMSE={result["RMSE"]:.3e}, '
        #         f'MAPE={result["MAPE"]:.2%}, '
        #         f'wMAPE={result["wMAPE"]:.2%}, '
        #         f'sMAPE={result["sMAPE"]:.2%}, '
        #         f'Acc2={result["ACC2"]:.2%}, '
        #         f'Acc3={result["ACC3"]:.2%}, '
        #         f'Acc4={result["ACC4"]:.2%}, '
        #         f'Complexity={result["complexity"]}, '
        #         f'Time={result["time"]:.2f} s)')
            
        #     # if self.config.early_stop and (self.config.get('ObsSNR', None) or self.config.get('spurious_link_ratio', None) or self.config.get('missing_link_ratio', None) or self.config.get('diff_as_out', None)):
        #     #     for _ in range(10):
        #     #         reward2, coef_list2, coef_dict2 = self.rewarder.solve(state, var_dict, sample=False)
        #     #         result2 = self.rewarder.evaluate(state, {**var_dict, **coef_dict2}, coef_list2)
        #     #         if result2['ACC4'] > 0.8: break
        #     #     logger.note(f'reward={reward2:.2f}, '
        #     #         f'R2={result2["R2"]:.4f}, '
        #     #         f'RMSE={result2["RMSE"]:.3e}, '
        #     #         f'MAPE={result2["MAPE"]:.2%}, '
        #     #         f'wMAPE={result2["wMAPE"]:.2%}, '
        #     #         f'sMAPE={result2["sMAPE"]:.2%}, '
        #     #         f'Acc2={result2["ACC2"]:.2%}, '
        #     #         f'Acc3={result2["ACC3"]:.2%}, '
        #     #         f'Acc4={result2["ACC4"]:.2%}')
        #     #     result['early_stop'] = (result2['ACC4'] >= 0.8)

        #     self.best_result = result

        # return reward

    def plot(self, ax=None, save_path=None, title=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3.26, 3.55), dpi=300)
        plt.rcParams['font.size'] = 7
        reward, best_reward, _ = zip(*self.search_history)
        middle = np.array([np.mean(r) for r in reward]) # (n_episodes, n_samples)
        upper = np.array([np.percentile(r, 100) for r in reward]) # (n_episodes, )
        lower = np.array([np.percentile(r, 0) for r in reward]) # (n_episodes, )
        best_reward = np.array(best_reward) # (n_episodes, )
        ax.plot(middle, color='gray', linewidth=1)
        ax.fill_between(np.arange(len(reward)), lower, upper, color='gray', alpha=0.2)
        ax.plot(best_reward, color='red', linewidth=2)
        if title: ax.set_title(title, fontsize=5)
        ax.set_xlim([0, len(reward)])
        ax.set_ylim([0, 1])
        if save_path: 
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)

    def Pareto(self, drop_non_pareto=True):
        """ 计算 R2 关于 Complexity 的 Pareto 前沿 """
        tmp = []
        for prefix, reward in tqdm(self.rewards.items()):
            if reward <= 0: continue
            complex = len(prefix)
            formula = GDExpr.prefix2str(list(prefix))
            # r_MSE = self.rewarder.config.complexity_base ** len(prefix) / reward - 1
            # R2 = 1 - r_MSE
            # RMSE = np.sqrt(r_MSE * self.var_dict['out'].var())
            R2 = self.rewarder.reward2R2(reward, prefix, self.var_dict)
            RMSE = self.rewarder.reward2RMSE(reward, prefix, self.var_dict)
            tmp.append([formula, complex, R2, RMSE, reward])
        
        df = pd.DataFrame(tmp, columns=['Formula', 'Complexity', 'R2', 'RMSE', 'Reward'])
        df = df.sort_values('R2', ascending=False).groupby('Complexity').head(1)
        df = df.sort_values('Complexity')
        while not (np.diff(df['R2'], prepend=0) > 0).all() and drop_non_pareto:
            df = df[np.diff(df['R2'], prepend=0) > 0]
        df.reset_index(drop=True, inplace=True)
        return df
