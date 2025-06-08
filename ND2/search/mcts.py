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
from ..GDExpr import GDExprClass, GDExpr
from .reward_solver import RewardSolver
from ..utils import NamedTimer, AbsTimer, Timer, seed_all
logger = logging.getLogger('ND2.MCTS')

class MCTS(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, 
                 rewarder:RewardSolver, 
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
                 log_detailed_speed:bool=False,
                 tempreture:float=0.0,
                 beam_size:int=10,
                 lambda_rollout:float=0.5,
                 max_token_num:int=30,
                 max_coeff_num:int=5,
                 random_state:int=0,
                 c_puct:float=5.0,
                 repeat_times:int=10,
                 use_random_simulate:bool=False,
                 **kwargs): 
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
        self.rewarder = rewarder
        self.ndformer = ndformer
        self.vars_node = vars_node
        self.vars_edge = vars_edge
        self.binary = binary
        self.unary = unary
        self.constant = constant
        self.log_per_episode = log_per_episode
        self.log_per_second = log_per_second
        self.log_detailed_speed = log_detailed_speed
        self.tempreture = tempreture
        self.beam_size = beam_size
        self.lambda_rollout = lambda_rollout
        self.max_token_num = max_token_num
        self.max_coeff_num = max_coeff_num
        self.random_state = random_state or np.random.randint(1, 35167)
        self.c_puct = c_puct
        self.repeat_times = repeat_times
        self.use_random_simulate = use_random_simulate
        
        self.tabula_rasa = (ndformer is None)
        self.actions = vars_node + vars_edge + binary + unary + constant + ['<C>']
        self.action2idx = {action: idx for idx, action in enumerate(self.actions)}
        self.n_actions = len(self.actions)

        if kwargs: logger.warning(f'Unused arguments: {kwargs} in MCTS')
        assert tempreture >= 0
        assert beam_size >= 1
        assert 0 <= lambda_rollout <= 1
        assert len(set(self.actions)) == len(self.actions)

        self.rewards = dict() # tuple(terminal prefix) -> reward
        self.MC_Tree = dict() # tuple(prefix) -> tuple(Q, N, P, V)
                              # Q: max possible reward of each child
                              # N: visit count of each child
                              # P: prior probability of each child
                              # V: prior value (or reward value if terminal) of current state
        self.best_model = []
        self.best_metric = defaultdict(lambda: np.nan) | {'reward': -np.inf}
        # self.search_history = [] # List of tuple([current rewards], current best reward, current best prefix)

        self.named_timer = NamedTimer()
        self.episode_timer = Timer(unit='episode')
        self.state_timer = Timer(unit='expanded node')
        self.eq_timer = AbsTimer(unit='eq')
        self.eq_timer.last = 0

    def fit(self, 
            root_prefix:List[str]=['node'], 
            episode_limit:int=1_000_000,
            time_limit:int=None,
            early_stop=lambda best_metric: best_metric['ACC4'] > 0.99):
        """
        Arguments:
        - root_prefix: e.g.1: ['node'], e.g.2: ['add', 'node', 'aggr', 'sour', 'node']
        """
        self.start_time = time.time()
        self.named_timer.clear()
        self.episode_timer.clear()
        self.state_timer.clear()
        self.eq_timer.clear()
        seed_all(self.random_state)

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
            self.eq_timer.add(len(self.rewards))
            # self.search_history.append((Q.tolist(), self.best_metric['reward'], self.best_metric['prefix']))
            if self.log_per_episode and (episode % self.log_per_episode == 0) or \
               self.log_per_second and (self.episode_timer.time > self.log_per_second):
                log = {
                    'Episode': f'{episode}',
                    'Best-RMSE': f'{self.best_metric["RMSE"]:.4f}',
                    'Best-R2': f'{self.best_metric["R2"]:.4f}',
                    'Best-Complexity': f'{self.best_metric["complexity"]}',
                    'Best-Equation': GDExpr.prefix2str(self.best_model) if len(self.best_model) else 'None',
                    'Search-Speed': f'{self.episode_timer}, {self.state_timer}, {self.eq_timer}',
                    # **self.best_metric,
                    'Time-Usage': str(self.named_timer),
                    # 'Current-Equation': GDExpr.prefix2str(states_to_expand[0]) if states_to_expand else 'None',
                }
                if not self.log_detailed_speed:
                    log.pop('Search-Speed')
                    log.pop('Time-Usage')
                self.named_timer.clear()
                self.episode_timer.clear()
                self.state_timer.clear()
                self.eq_timer.clear()
                logger.info(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))

            if early_stop is not None and early_stop(self.best_metric):
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
        beam_search = [ 
            ([(None, root_prefix)], [], False) # route, UCT_route, Is_leaf_or_terminal
        ] 
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
            metric = np.concatenate([sum(UCT_route) / len(UCT_route) for UCT_route in UCT_routes], axis=0)
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
                    UCT_route = [*UCT_routes[idx][:-1], UCT_routes[idx][-1][action:action+1]]
                    done = (tuple(state) not in self.MC_Tree) or (not self.get_mask(state).any())
                    _beam_search.append(([*route, (self.actions[action], state)], UCT_route, done))
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
            return self.neural_simulate(states_to_expand)

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

    def neural_simulate(self, states_to_expand:List[List[str]]):
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
        if self.max_token_num - token_num >= 2: mask[vectorize(self.binary)] = True
        # Unary operators
        if self.max_token_num - token_num >= 1: mask[vectorize(self.unary)] = True
        if type == 'edge': 
            if 'aggr' in self.actions: mask[vectorize('aggr')] = False
            if 'rgga' in self.actions: mask[vectorize('rgga')] = False
        if type == 'node':
            if 'sour' in self.actions: mask[vectorize('sour')] = False
            if 'targ' in self.actions: mask[vectorize('targ')] = False
        if self.max_token_num - token_num >= 0:
            # Variables
            if type == 'node': mask[vectorize(self.vars_node)] = True
            if type == 'edge': mask[vectorize(self.vars_edge)] = True
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
                mask[vectorize(self.constant)] = True
                if prefix.count('<C>') < self.max_coeff_num:
                    mask[vectorize('<C>')] = True
        return mask

    def get_UCT(self, state:List[str]) -> np.ndarray:
        Q, N, P = self.MC_Tree[tuple(state)]
        if self.tabula_rasa:
            UCT = Q + self.c_puct * np.sqrt(np.log(N.sum().clip(1)) / N.clip(1))
        else:
            UCT = Q + self.c_puct * np.sqrt(1 + N.sum()) * P / (1 + N)
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
    
    def get_rewards(self, states):
        rewards = np.array([self.get_reward(state)[0] for state in states])

        if any(rewards >= self.best_metric['reward']):
            for idx, state in enumerate(states):
                if rewards[idx] <= self.best_metric['reward']: continue
                # Recheck the reward
                rewards[idx], prefix_with_coef = self.get_reward(state, use_cache=False, update_cache=True, sample=False)
                if prefix_with_coef is None: continue
                if rewards[idx] <= self.best_metric['reward']: continue
                # Update the best result
                self.best_metric = dict(reward=rewards[idx], equation=GDExpr.prefix2str(prefix_with_coef), time=time.time()-self.start_time) | self.rewarder.evaluate(prefix_with_coef, {})
                self.best_model = prefix_with_coef
                log = self.best_metric
                logger.note('Update best result: ' + ' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))

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

        reward, prefix_with_coef = self.rewarder.solve(state, sample=sample, max_iter=30)
        if update_cache: self.rewards[tuple(state)] = reward
        return reward, prefix_with_coef

    def plot(self, ax=None, save_path=None, title=None):
        raise DeprecationWarning('This function is deprecated.')
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

    def Pareto(self, 
               topk=20,
               max_iter=100,
               accuracy_metric=lambda metric: metric['R2'],
               print_on_fly=True
               ):
        """ Calculate Pareto Front """
        logger.note('Calculating Pareto Front... This may take a while.')
        results = defaultdict(list)
        for prefix, reward in self.rewards.items():
            if not np.isfinite(reward) or reward <= 0: continue
            if len(prefix) > self.max_token_num: continue
            if prefix.count('<C>') > self.max_coeff_num: continue
            complex = len(prefix)
            results[complex].append((prefix, reward))
        # select topk among each complexity
        for complex, items in results.items():
            results[complex] = sorted(items, key=lambda x: x[1], reverse=True)[:topk]
        # sort by complex
        results = dict(sorted(results.items(), key=lambda x: x[0]))

        pareto = []
        cur_best_accuracy = -np.inf
        for complex, items in tqdm(results.items(), leave=False, dynamic_ncols=True, disable=print_on_fly):
            for idx, (prefix, reward) in enumerate(items):
                reward, prefix_with_coef = self.rewarder.solve(list(prefix), sample=False, max_iter=max_iter)
                metrics = self.rewarder.evaluate(prefix_with_coef, {})
                accuracy = accuracy_metric(metrics)
                results[complex][idx] = (prefix_with_coef, accuracy)
            prefix_with_coef, accuracy = max(results[complex], key=lambda x: x[1])
            if accuracy > cur_best_accuracy:
                pareto.append((prefix_with_coef, complex, accuracy))
                cur_best_accuracy = accuracy
                if print_on_fly: logger.note(f'Complex={complex:<5} Accuracy={accuracy:<10.5f} {GDExpr.prefix2str(prefix_with_coef)}')
            else:
                if print_on_fly: logger.debug(f'Complex={complex:<5} Accuracy={cur_best_accuracy:<10.5f} {GDExpr.prefix2str(prefix_with_coef)}')

        return pareto
