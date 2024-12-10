import numpy as np
import networkx as nx
from typing import List, Tuple
from ..GDExpr import GDExpr


class Generator(object):
    def __init__(self,
                 GMM=True,
                 min_data_num=100,
                 max_data_num=200,
                 min_node_num=10,
                 max_node_num=100,
                 min_edge_num=20,
                 max_edge_num=600,
                 min_var_val=-10,
                 max_var_val=10,
                 min_coeff_val=-20,
                 max_coeff_val=20):
        self.use_GMM = GMM
        self.min_data_num = min_data_num
        self.max_data_num = max_data_num
        self.min_node_num = min_node_num
        self.max_node_num = max_node_num
        self.min_edge_num = min_edge_num
        self.max_edge_num = max_edge_num
        self.min_var_val = min_var_val
        self.max_var_val = max_var_val
        self.min_coeff_val = min_coeff_val
        self.max_coeff_val = max_coeff_val

    def generate_graph(self, type:str, **kwargs):
        """
        Arguments:
        - V: node num
        - type: 'ER', 'BA', 'WS', 'Complete
        - kwargs:
            (When type is 'ER')
            - p: edge probability
            - directed: directed or not
            (When type is 'BA')
            - m: number of edges to attach from a new node to existing nodes
            (When type is 'WS')
            - k: each node is connected to k nearest neighbors in ring topology
            - p: probability of rewiring each edge
            (When type is 'Complete')
            - None
        Return:
        - A: (V, V), adjacency matrix
        - G: (E, 2), edge list
        - V: node num
        - E: edge num
        """
        if type == 'ER':
            V = kwargs.get('V', np.random.randint(self.min_node_num, self.max_node_num))
            E = kwargs.get('E', np.random.randint(self.min_edge_num, self.max_edge_num))
            p = E / (V * (V-1))
            directed = kwargs.get('directed', np.random.randint(0, 2))
            graph = nx.erdos_renyi_graph(V, p, directed=directed)
        elif type == 'BA':
            V = kwargs.get('V', np.random.randint(self.min_node_num, self.max_node_num))
            m = kwargs.get('m', np.random.choice([1, 2, 3]))
            graph = nx.barabasi_albert_graph(V, m)
        elif type == 'WS':
            V = kwargs.get('V', np.random.randint(self.min_node_num, self.max_node_num))
            k = kwargs.get('k', np.random.choice([2, 4, 6]))
            p = kwargs.get('p', np.random.uniform(0.1, 0.9))
            graph = nx.watts_strogatz_graph(V, k, p)
        elif type == 'Complete':
            V = kwargs.get('V', np.random.randint(self.min_node_num, min(self.max_node_num, int(np.sqrt(self.max_edge_num))) + 1))
            graph = nx.complete_graph(V)
        else:
            raise ValueError(f'Unknown graph type {type}')
        A = nx.to_numpy_array(graph, dtype=bool)
        G = np.stack(A.nonzero(), axis=-1)
        V = A.shape[0]
        E = G.shape[0]
        return A, G, V, E


    def generate_data(self, prefix:List[str], root_type:str):
        """
        Arguments:
        - prefix: a list of words
        - root_type: 'e' or 'edge'

        Returns:
            var_dict = dict(
                A: np.ndarray, (V, V)
                G: np.ndarray, (E, 2)
                out: np.ndarray, (N, V) or (N, E)
                v1/v2/v3/v4/v5: np.ndarray, (N, V)
                e1/e2/e3/e4/e5: np.ndarray, (N, E)
            )
        """
        assert root_type in ['node', 'edge'], root_type
        N = np.random.randint(self.min_data_num, self.max_data_num + 1)
        A, G, V, E = self.generate_graph(np.random.choice(['ER', 'BA', 'WS', 'Complete'], p=[0.3, 0.3, 0.3, 0.1]))
        # assert G.shape == (E, 2) and A.shape == (V, V)
        # assert G.max() < V and G.min() >= 0 and A.sum() == E
        node_vars = [var for var in ['v1', 'v2', 'v3', 'v4', 'v5'] if var in prefix]
        edge_vars = [var for var in ['e1', 'e2', 'e3', 'e4', 'e5'] if var in prefix]
        var_dict = dict(A=A, G=G)

        n_coef = prefix.count('<C>')
        coef_list = np.random.uniform(self.min_coeff_val, 
                                        self.max_coeff_val, (n_coef,))

        if self.use_GMM: # GMM Prior
            tmp = self.GMM(N*V, len(node_vars))
            for idx, var in enumerate(node_vars):
                var_dict[var] = tmp[:, idx].reshape(N, V)
            tmp = self.GMM(N*E, len(edge_vars))
            for idx, var in enumerate(edge_vars):
                var_dict[var] = tmp[:, idx].reshape(N, E)
        else: # Uniform Prior
            for var in node_vars:
                var_dict[var] = np.random.uniform(self.min_var_val, 
                                                    self.max_var_val, (N, V))
            for var in edge_vars:
                var_dict[var] = np.random.uniform(self.min_var_val, 
                                                    self.max_var_val, (N, E))

        var_dict['out'] = GDExpr.eval(prefix, var_dict, coef_list)
        return var_dict

    def GMM(self, N, D, L=1):
        K = np.random.randint(1, 11)
        pi = np.random.rand(K); pi /= pi.sum()
        sigma_Z = np.random.uniform(0.0, 10.0, (K,))
        sigma_X = np.random.uniform(0.0, 3.0, (K,))
        A = np.random.uniform(-1.0, 1.0, (D, L, K))
        b = np.random.uniform(-10.0, 10.0, (D, K))
        C = np.random.choice(K, (N,), p=pi)
        Z = np.random.normal(0, sigma_Z, (N, L, K))
        n = np.random.normal(0, sigma_X, (N, D, K))
        X = np.einsum('DlK,NlK->NDK', A, Z) + b + n
        return np.choose(C, X.transpose(2, 1, 0)).transpose(1, 0)
