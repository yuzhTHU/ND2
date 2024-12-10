import os
import sys
import torch
import pickle
import logging
import numpy as np
import torch.utils.data as D
from copy import deepcopy
from typing import List
from ..GDExpr import GDExpr, is_float
from .generator import Generator


logger = logging.getLogger('ND2.dataset')

class Dataset(D.Dataset):
    def __init__(self, root_types:List[str], 
                 prefixes:List[List[str]], datafiles:List[str],
                 n_expr_sample=np.inf, data_dir='./data/synthetic/',
                 data_only=False, device='cpu'):
        """
        Arguments:
        - root_types: List[str], len=M
        - prefixes: List[List[str]], len=M
        - datafiles: List[str|None], len=M
        - n_expr_sample: int
            sample expressions from each decomposed prefix
        - n_data_sample: int
            sample data from each datafile (N*V or N*E)
        - data_only: bool
            return tuple(var_dict, root_type, prefix) if true
            return tuple(var_dict, root_type, prefixes, policies, indexes, parents, types) if false
        """
        super().__init__()
        self.root_types = root_types
        self.prefixes = prefixes
        self.datafiles = datafiles
        self.n_expr_sample = n_expr_sample
        self.data_only = data_only
        self.data_dir = data_dir
        self.device = device

    def __len__(self):
        return len(self.prefixes)
    
    def __getitem__(self, idx):
        """
        Returns:
        - var_dict: Dict[A, G, out, variables...]
        - root_type: 'node' or 'edge'
        - prefix: List[torch.LongTensor(L)], len=M (M <= L)
            start with <SOS>, end with <EOS>,
            including a placeholder after the starting <SOS>
        - value: None
        - policy: torch.LongTensor(M,)
        - index: torch.LongTensor(M,)
        - parents: List[torch.LongTensor(L)], len=M
        - types: List[torch.LongTensor(L)], len=M
        """

        root_type = deepcopy(self.root_types[idx])
        prefix = deepcopy(self.prefixes[idx])
        datafile = deepcopy(self.datafiles[idx])

        # data
        if datafile is not None: var_dict = self.load_data(datafile)
        else:                    var_dict = self.generate_data(root_type, prefix)

        for i, item in enumerate(prefix):
            if is_float(item) and str(item) not in GDExpr.word2id: prefix[i] = GDExpr.coeff_token

        if self.data_only:
            return var_dict, root_type, prefix # ignore value, policy, index, parent, type

        # prefixes, policies, indexes, parents, types
        prefixes, policies, indexes, parents, types = [], [], [], [], []
        sample_num = min(self.n_expr_sample, len(prefix))
        sample_idx = np.sort(np.random.choice(len(prefix), sample_num, replace=False))
        for i in range(len(prefix)):
            prefix, policy, index = GDExpr.decompose(prefix, root_type)
            if i not in sample_idx: continue
            prefixes.append(torch.LongTensor(GDExpr.vectorize([
                'sos', root_type, *prefix, 'eos'])).to(self.device))
            policies.append(GDExpr.word2id[policy])
            indexes.append(index + 1) # prevent 0, start from 1, same as parents, where 0 means pad
            parents.append(torch.LongTensor([
                0, 0, *GDExpr.analysis_parent(prefix, 0, 1), 0]).to(self.device))
            types.append(torch.LongTensor(GDExpr.vectorize([
                'sos', root_type, *GDExpr.analysis_type(prefix, root_type), 'eos'])).to(self.device))
        policies = torch.LongTensor(policies).to(device=self.device) # (M,)
        indexes = torch.LongTensor(indexes).to(device=self.device) # (M,)
        return var_dict, root_type, prefixes, None, policies, indexes, parents, types

    def generate_data(self, root_type, prefix):
        generator = Generator()
        var_dict = generator.generate_data(prefix, root_type)
        valid = np.isfinite(var_dict['out'])
        N, V, E = var_dict['out'].shape[0], var_dict['A'].shape[0], var_dict['G'].shape[0]
        logger.debug(f'Generate data with {N} samples, {V} nodes, {E} edges, '
                        f'Out Range {np.nanmin(var_dict["out"]):.2f}~{np.nanmax(var_dict["out"]):.2f}, '
                        f'NaN Rate {np.isnan(var_dict["out"]).mean():.2%}, '
                        f'Valid Rate {valid.mean():.2%}'
                        f'[{GDExpr.prefix2str(prefix)}]')
        return var_dict

    def load_data(self, datafile):
        if '--use_json' in sys.argv:
            import json
            path = os.path.join(self.data_dir, f'{datafile}.json')
            var_dict = json.load(open(path, 'r'))
            for k, v in var_dict.items():
                var_dict[k] = np.array(v)
        else:
            path = os.path.join(self.data_dir, f'{datafile}.pkl')
            var_dict = pickle.load(open(path, 'rb'))
        logger.debug(f'Load data from {path}')
        # TODO: 添加 reindex 之类的功能
        return var_dict

    @classmethod
    def load_csv(cls, filenames, **kwargs):
        if isinstance(filenames, str): filenames = [filenames]
        root_types = []
        prefixes = []
        datafiles = []
        for filename in filenames:
            with open(filename, 'r') as f:
                for line in f:
                    prefix = line.rstrip('\n').split(',')
                    root_type = prefix.pop(0)
                    if root_type == '<V>': root_type = 'node'
                    if root_type == '<E>': root_type = 'edge'
                    if prefix[-1].startswith('@'):
                        datafile = prefix.pop().lstrip('@')
                    else:
                        datafile = None
                    for token in prefix:
                        if token not in GDExpr.word2id and not is_float(token):
                            break
                    else:
                        root_types.append(root_type)
                        prefixes.append(prefix)
                        datafiles.append(datafile)
        return cls(root_types, prefixes, datafiles, **kwargs)
