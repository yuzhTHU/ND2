import sys
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union, Literal
from .utils import GNN, PositionalEncoding
from ..GDExpr import GDExpr
import warnings


# See https://github.com/pytorch/pytorch/issues/100469
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.")
logger = logging.getLogger('ND2.Model')


class Encoder(nn.Module):
    def __init__(self, d_emb, dropout, d_data_feat, n_node_vars, n_edge_vars,
                 n_transformer_layers, n_GNN_layers, max_sample_num, split, **kwargs):
        super().__init__()
        self.d_emb = d_emb
        self.d_ff = d_emb * 4
        self.n_head = d_emb // 64
        self.dropout = dropout
        self.d_data_feat = d_data_feat
        self.max_node_vars_n = n_node_vars
        self.max_edge_vars_n = n_edge_vars
        self.n_transformer_layers = n_transformer_layers
        self.n_GNN_layers = n_GNN_layers
        self.max_sample_num = max_sample_num
        self.split = split
        if kwargs: logger.warning(f'Unused kwargs: {kwargs} in Encoder')

        self.GNN = GNN(self.d_emb, 
                       self.n_GNN_layers, 
                       self.max_node_vars_n * self.d_data_feat,
                       self.max_edge_vars_n * self.d_data_feat, 
                       self.dropout)
        # self.norm = nn.LayerNorm(self.d_emb)
        self.Transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model=self.d_emb, 
                nhead=self.n_head, 
                dim_feedforward=self.d_ff, 
                dropout=self.dropout, 
                batch_first=True
            ), self.n_transformer_layers)

    def forward(self, v_bits, e_bits, G, A, root_type, mask=None):
        """
        BatchSize = 1!!!
        Input:
        - v_bits: (N, V, <= max_node_vars_n, d_data_feat=16), multi-hot embedding
        - e_bits: (N, E, <= max_edge_vars_n, d_data_feat=16), multi-hot embedding
        - G: (E, 2) (type=int64)
        - A: (V, V) (type=int64)
        - root_type: 'node' or 'edge'
        - mask: (N, V or E), only smaple those mask=True

        Output:
        - out: (max_sample_num, d_emb)
        """
        v_bits = F.pad(v_bits, (0, 0, 0, self.max_node_vars_n - v_bits.shape[2]), value=0.) # (N, V, max_node_vars_n, 16)
        e_bits = F.pad(e_bits, (0, 0, 0, self.max_edge_vars_n - e_bits.shape[2]), value=0.) # (N, E, max_edge_vars_n, 16)
        v_emb = v_bits.flatten(-2, -1) # (N, V, max_node_vars_n * d_data_feat)
        e_emb = e_bits.flatten(-2, -1) # (N, E, max_edge_vars_n * d_data_feat)
        
        tmp = []
        if self.split:
            for n in range(v_bits.shape[0]):
                tmp.append(self.GNN(v_emb[(n,), ...], e_emb[(n,), ...], G, A)) # (N, V, d_emb), (N, E, d_emb)
            v_emb, e_emb = zip(*tmp)
            v_emb, e_emb = torch.concatenate(v_emb, axis=0), torch.concatenate(e_emb, axis=0)
        else:
            v_emb, e_emb = self.GNN(v_emb, e_emb, G, A) # (N, V, d_emb), (N, E, d_emb)
        data_emb = v_emb if root_type == 'node' else e_emb # (N, V/E, d_emb)
        if mask is not None:
            data_emb = data_emb[mask, :] # (subset of {N * V/E}, d_emb)
        else:
            data_emb = data_emb.flatten(0, 1) # (N * V/E, d_emb)
        if data_emb.shape[0] > self.max_sample_num:
            data_emb = data_emb[torch.randperm(data_emb.shape[0])[:self.max_sample_num]] # (N_max, d_emb)
        data_emb = self.Transformer(data_emb)
        return data_emb


class Decoder(nn.Module):
    def __init__(self, use_aux_input, n_words, d_emb, dropout,
                 n_transformer_layers, max_seq_len):
        super().__init__()
        self.use_aux_input = use_aux_input
        self.d_emb = d_emb
        self.d_ff = d_emb * 4
        self.n_head = d_emb // 64
        self.dropout = dropout
        self.n_words = n_words
        self.n_decoder_layers = n_transformer_layers
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(self.n_words, self.d_emb)
        self.index_embedding = nn.Embedding(self.max_seq_len, self.d_emb)
        self.pe = PositionalEncoding(self.d_emb, self.dropout)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
                d_model=self.d_emb, 
                nhead=self.n_head, 
                dim_feedforward=self.d_ff, 
                batch_first=True, 
                dropout=self.dropout
            ), self.n_decoder_layers)
        self.value_head = nn.Linear(self.d_emb, 1)
    
    def forward(self, data_emb, expr_ids, parents=None, types=None):
        """
        Input:
        - data_emb: (N, d_emb) or (M, N, d_emb)
        - expr_ids: (M, L) (type=int64), padded with <PAD>, start with <SOS>, end with <SOS>
        - parents: (M, L) (type=int64), padded with <PAD>, start with <SOS>, end with <SOS>
        - types: (M, L) (type=int64), padded with <PAD>, start with <SOS>, end with <SOS>

        Output:
        - value: (M, 1)
        - policy: (M, n_words)
        - index: (M, max_seq_len)
        """
        expr_ids = F.pad(expr_ids, (1, 0), value=GDExpr.word2id.query_index) # (M, 1+L)
        expr_ids = F.pad(expr_ids, (1, 0), value=GDExpr.word2id.query_policy) # (M, 2+L)
        expr_ids = F.pad(expr_ids, (1, 0), value=GDExpr.word2id.query_value) # (M, 3+L)
        formula_emb = self.token_embedding(expr_ids) # (M, 3+L, d_emb)
        if self.use_aux_input and parents is not None: 
            formula_emb[:, 3:, :] = formula_emb[:, 3:, :] + self.index_embedding(parents) # (M, 3+L, d_emb)
        if self.use_aux_input and types is not None:   
            formula_emb[:, 3:, :] = formula_emb[:, 3:, :] + self.token_embedding(types) # (M, 3+L, d_emb)
        formula_emb[:, 3:, :] = self.pe(formula_emb[:, 3:, :]) # (M, 3+L, d_emb)

        if data_emb.dim() == 2:
            data_emb = data_emb.unsqueeze(0).expand(expr_ids.shape[0], -1, -1) # (M, N, d_emb)
        assert data_emb.dim() == 3 and data_emb.shape[0] == expr_ids.shape[0] # (M, N, d_emb)

        # make sure all tokens cannot see <PAD>
        key_padding_mask = (expr_ids == GDExpr.pad_id) # (M, 3+L)
        # make sure tokens cannot see query_[value/policy/index] while the latter can see former and theirself
        tgt_mask = torch.full((expr_ids.shape[1], expr_ids.shape[1]), 
                              fill_value=False, device=expr_ids.device) # (3+L, 3+L)
        tgt_mask[:, :3] = True
        tgt_mask[:3, :3].diagonal()[:] = False
        out = self.decoder(formula_emb, data_emb, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=key_padding_mask) # (M, 3+L, d_emb)

        value = self.value_head(out[:, 0, :]).squeeze(-1) # (M,)
        policy = F.linear(out[:, 1, :], self.token_embedding.weight) # (M, n_words)
        index = F.linear(out[:, 2, :], self.index_embedding.weight) # (M, max_seq_len)
        out = out[:, 3:, :] # (M, L, d_emb)

        return value, policy, index


class NDformer(nn.Module):
    def __init__(self, 
                 d_emb=512,
                 dropout=0.1,
                 d_data_feat=16, # sign bit (1) + exponent bits (5) + mantissa bits (10)
                 n_node_vars=6,
                 n_edge_vars=6,
                 n_transformer_encoder_layers=2,
                 n_GNN_layers=2,
                 max_sample_num=3000,
                 split=False,
                 use_aux_input=True,
                 n_words=60, # vocabulary size
                 n_transformer_decoder_layers=2,
                 max_seq_len=100,  # no less than GDExpr.max_complexity + 3
                 device='cpu'):
        super().__init__()
        self.encoder = Encoder(d_emb=d_emb, 
                               dropout=dropout, 
                               d_data_feat=d_data_feat, 
                               n_node_vars=n_node_vars, 
                               n_edge_vars=n_edge_vars, 
                               n_transformer_layers=n_transformer_encoder_layers, 
                               n_GNN_layers=n_GNN_layers, 
                               max_sample_num=max_sample_num,
                               split=split)
        self.decoder = Decoder(use_aux_input=use_aux_input,
                               d_emb=d_emb, 
                               dropout=dropout, 
                               n_words=n_words,
                               n_transformer_layers=n_transformer_decoder_layers,
                               max_seq_len=max_seq_len)

        self.optimizer = None
        self.scheduler = None
        self.device = device
        self.to(self.device)
    
    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def save(self, path, optimizer=None, scheduler=None):
        torch.save(dict(
            encoder=self.encoder.state_dict(),
            decoder=self.decoder.state_dict(),
            optimizer=optimizer.state_dict() if optimizer is not None else None,
            scheduler=scheduler.state_dict() if scheduler is not None else None,
        ), path)
    
    def load(self, path, optimizer=None, scheduler=None, weights_only=False):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=weights_only)
        except ModuleNotFoundError as e:
            # 这是将 src 重命名为 ND2 后导致的问题
            from io import BytesIO
            content = open(path, 'rb').read()
            content = content.replace(b'src', b'ND2')
            ckpt = torch.load(BytesIO(content), map_location=self.device, weights_only=weights_only)

        try:
            self.encoder.load_state_dict(ckpt['encoder'])
            self.decoder.load_state_dict(ckpt['decoder'])
            if optimizer is not None: 
                if ckpt['optimizer'] is not None: 
                    optimizer.load_state_dict(ckpt['optimizer'])
                else: logger.warning('Optimizer is not saved in checkpoint!')
                    
            if scheduler is not None:
                if ckpt['scheduler'] is not None: 
                    scheduler.load_state_dict(ckpt['scheduler'])
                else: logger.warning('Scheduler is not saved in checkpoint!')
        except Exception as e:
            logger.warning(f'Load checkpoint "{path}" failed: {e}')
        
    def encode(self, root_type, var_dict, mask=None, sample=True):
        """
        Input:
        - v: (N, V, <= max_node_vars_n)
        - e: (N, E, <= max_edge_vars_n)
        - G: (E, 2) (type=int64)
        - A: (V, V) (type=int64)
        - root_type: 'node' or 'edge'

        Output:
        - data_emb: (N_sample, d_emb)
        """
        A = var_dict['A'] # (V, V)
        G = var_dict['G'] # (E, 2)
        out = var_dict['out'] # (N, V) or (N, E)
        N, V, E = out.shape[0], A.shape[0], G.shape[0]
        v = [out if root_type == 'node' else np.zeros((N, V))] + \
            [var_dict.get(var, np.zeros((N, V))) for var in GDExpr.variable.node]
        v = np.stack(v, axis=-1) # (N, V, 1 + d_v)
        e = [out if root_type == 'edge' else np.zeros((N, E))] + \
            [var_dict.get(var, np.zeros((N, E))) for var in GDExpr.variable.edge]
        e = np.stack(e, axis=-1) # (N, E, 1 + d_e)

        if sample:
            VorE = (V if root_type == 'node' else E)
            n = int(np.ceil(3 * self.encoder.max_sample_num / VorE))
            if n < N:
                sample_idx = np.random.choice(N, n, replace=False)
                logger.debug(f'[Encoder] Sample data: ({N}*{VorE}) -> ({n}*{VorE})')
                v = v[sample_idx]
                e = e[sample_idx]
                N = n
                if mask is not None: mask = mask[sample_idx]

        v_bits = torch.from_numpy(GDExpr.parse_float(v)).to(self.device, torch.float32) # (N, V, 1 + d_v, 16)
        e_bits = torch.from_numpy(GDExpr.parse_float(e)).to(self.device, torch.float32) # (N, E, 1 + d_e, 16)
        G = torch.from_numpy(G).to(self.device, torch.long) # (E, 2)
        A = torch.from_numpy(A).to(self.device, torch.long) # (V, V)
        data_emb = self.encoder(v_bits, e_bits, G, A, root_type, mask=mask)
        return data_emb

    def decode(self, data_emb:torch.Tensor, expr_ids:torch.LongTensor, parents:torch.LongTensor=None, types:torch.LongTensor=None):
        """
        Input:
        - data_emb: (N_sample, d_emb)
        - expr_ids: (M, max_len) (type=int64), 
            start with <SOS>, end with <EOS>, padded with <PAD>,
            a placeholder following starting <SOS>
        - parents: (M, max_len), (type=int64), start / end / pad same as expr_ids
        - types: (M, max_len), (type=int64), start / end / pad same as expr_ids

        Output:
        - value: (M,)
        - policy: (M, n_words)
        - index: (M, max_seq_len)
        """
        data_emb = data_emb.to(self.device)
        expr_ids = expr_ids.to(self.device)
        parents = parents.to(self.device)
        types = types.to(self.device)
        value, policy, index = self.decoder(data_emb, expr_ids, parents, types)
        return value, policy, index

    def set_data(self, 
                 Xv:Dict[str, np.ndarray], 
                 Xe:Dict[str, np.ndarray], 
                 A:np.ndarray, 
                 G:np.ndarray, 
                 Y:np.ndarray, 
                 root_type:Literal['node', 'edge'], 
                 mask=None,
                 cache_data_emb=True):
        """
        Input:
        - Xv: Dict of str -> (N, V)
        - Xe: Dict of str -> (N, E)
        - A: (V, V) (int)
        - G: (E, 2) (int)
        - Y: (N, V or E)
        - root_type: 'node' or 'edge'
        - mask: (N, V or E)
        """
        V = A.shape[0]
        E = G.shape[0]
        T = Y.shape[0]
        assert A.shape == (V, V)
        assert G.shape == (E, 2)
        assert (Y.shape == (T, V) if root_type == 'node' else (T, E))
        
        var_dict = dict(A=A, G=G, out=Y)
        for idx, (k, v) in enumerate(Xv.items(), 1):
            assert v.shape in [(T, V), (1, V), (V,), (T, 1)]
            if v.ndim == 1: v = v.reshape(1, -1).repeat(T, axis=0)
            if v.shape[-1] == 1: v = v.repeat(V, axis=-1)
            var_dict[f'v{idx}'] = v
        for idx, (k, e) in enumerate(Xe.items(), 1):
            assert e.shape in [(T, E), (1, E), (E,), (T, 1)]
            if e.ndim == 1: e = e.reshape(1, -1).repeat(T, axis=0)
            if e.shape[-1] == 1: e = e.repeat(E, axis=-1)
            var_dict[f'e{idx}'] = e
        self.var_dict = var_dict
        self.var_map = {k: f'v{i}' for i, k in enumerate(Xv, 1)} | \
                       {k: f'e{i}' for i, k in enumerate(Xe, 1)}
        self.i_var_map = {v: k for k, v in self.var_map.items()}
        self.root_type = root_type
        self.cache_data_emb = cache_data_emb
        self.data_emb = None
        if cache_data_emb:
            with torch.no_grad():
                self.data_emb = self.encode(self.root_type, self.var_dict, mask)

    def get_policy(self, 
                   prefixes:List[List[str]], 
                   actions:List[str]=None,
                   mask:List[np.ndarray]=None):
        self.eval()
        with torch.no_grad():
            if not hasattr(self, 'var_dict'): raise ValueError('Please call .set_data() first!')
            
            prefixes = [[self.var_map.get(token, token) for token in prefix] for prefix in prefixes]

            if not self.cache_data_emb:
                self.data_emb = self.encode(self.root_type, self.var_dict)
            data_emb = self.data_emb.to(self.device)

            expr_ids = [torch.from_numpy(GDExpr.vectorize(['sos', self.root_type, *prefix, 'eos'])) for prefix in prefixes]
            expr_ids = torch.nn.utils.rnn.pad_sequence(expr_ids, batch_first=True, padding_value=GDExpr.pad_id)
            expr_ids = expr_ids.to(self.device)

            parents = [torch.LongTensor([0, 0, *GDExpr.analysis_parent(prefix, 0, 1), 0]) for prefix in prefixes]
            parents = torch.nn.utils.rnn.pad_sequence(parents, batch_first=True, padding_value=GDExpr.pad_id)
            parents = parents.to(self.device)

            types = [torch.from_numpy(GDExpr.vectorize(['sos', self.root_type, *GDExpr.analysis_type(prefix, self.root_type), 'eos'])) for prefix in prefixes]
            types = torch.nn.utils.rnn.pad_sequence(types, batch_first=True, padding_value=GDExpr.pad_id)
            types = types.to(self.device)

            with torch.no_grad():
                _, policy, _ = self.decoder(data_emb, expr_ids, parents, types)
            if actions is not None:
                policy = policy[:, GDExpr.vectorize([self.var_map.get(a, a) for a in actions])]
            if mask is not None:
                mask = torch.from_numpy(np.stack(mask, axis=0)).to(self.device)
                policy.masked_fill_(~mask, -np.inf)
            policy = policy.softmax(-1).cpu().numpy()

        return policy

    def forward(self, prefixes:List[List[str]], root_type:str, 
                var_dict:dict=None, data_emb:torch.tensor=None, 
                valid_policy:List[np.ndarray]=None):
        """
        Input:
        - prefixes: List of prefix, without <SOS> or <PAD>, len=M
        - root_type: 'node' or 'edge'
        - var_dict or data_emb
            - var_dict = Dict(out, G, A, variables...)
            - data_emb = Tensor(N, d_emb)
        - valid_policy: list of np.ndarray, shape=(n_words,), bool, cannot be all False for any prefix

        Output:
        - value: (M,)
        - policy: (M, n_words), softmaxed
        - index: (M, max_seq_len), softmaxed
        """
        assert (var_dict is None) ^ (data_emb is None)
        with torch.no_grad():
            if data_emb is None: data_emb = self.encode(root_type, var_dict=var_dict)
            expr_ids = [torch.from_numpy(GDExpr.vectorize(['sos', root_type, *prefix, 'eos'])) for prefix in prefixes]
            parents = [torch.LongTensor([0, 0, *GDExpr.analysis_parent(prefix, 0, 1), 0]) for prefix in prefixes]
            types = [torch.from_numpy(GDExpr.vectorize(['sos', root_type, *GDExpr.analysis_type(prefix, root_type), 'eos'])) for prefix in prefixes]
            expr_ids = torch.nn.utils.rnn.pad_sequence(expr_ids, batch_first=True, padding_value=GDExpr.pad_id)
            parents = torch.nn.utils.rnn.pad_sequence(parents, batch_first=True, padding_value=GDExpr.pad_id)
            types = torch.nn.utils.rnn.pad_sequence(types, batch_first=True, padding_value=GDExpr.pad_id)
            data_emb = data_emb.to(self.device)
            expr_ids = expr_ids.to(self.device)
            parents = parents.to(self.device)
            types = types.to(self.device)
            value, policy, index = self.decoder(data_emb, expr_ids, parents, types)
            if valid_policy is not None:
                valid_policy = torch.from_numpy(np.stack(valid_policy, axis=0))
                valid_policy = valid_policy.to(self.device)
                policy.masked_fill_(~valid_policy, -np.inf)
            valid_index = torch.full(index.shape, fill_value=False, device=index.device)
            valid_index[:, :expr_ids.shape[1]] = (expr_ids == GDExpr.placeholder.node) | (expr_ids == GDExpr.placeholder.edge)
            index.masked_fill_(~valid_index, -np.inf)
        return value, policy.softmax(-1), index.softmax(-1)        
