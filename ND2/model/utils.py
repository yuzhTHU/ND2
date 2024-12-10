import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log


class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, residual=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.residual = residual
        self.update_node = nn.Sequential(nn.Linear(node_dim * 2 + edge_dim, out_dim),
                                         nn.ReLU(),
                                         nn.Linear(out_dim, out_dim))
        self.update_edge = nn.Sequential(nn.Linear(node_dim * 2 + edge_dim, out_dim),
                                         nn.ReLU(),
                                         nn.Linear(out_dim, out_dim))
        if residual:
            self.shortcut_node = nn.Linear(node_dim, out_dim) \
                                if node_dim != out_dim else nn.Identity()
            self.shortcut_edge = nn.Linear(edge_dim, out_dim) \
                                if edge_dim != out_dim else nn.Identity()

    def forward(self, v, e, G, A):
        """
        - v: [N, V, node_dim]
        - e: [N, E, edge_dim]
        - G: [E, 2]
        - A: [V, V]
        - return: [N, V, out_dim], [N, E, out_dim]
        """
        N, V, _ = v.shape
        _, E, _ = e.shape
        v_sour = v[:, G[:, 0], :] # [N, E, node_dim]
        v_term = v[:, G[:, 1], :] # [N, E, node_dim]
        x = torch.cat([v_sour, v_term, e], dim=-1)

        y = self.update_node(x) # [N, E, out_dim]
        update_v = torch.zeros(N, V, self.out_dim, device=y.device)
        update_v.scatter_add_(dim=1, index=G[None, :, (1,)].expand(N, -1, self.out_dim), src=y)
        degree = A.sum(dim=0).detach().clamp(1) # [V,]
        # degree = torch.ones(V, device=A.device) # [V,]
        update_v = update_v / degree[None, :, None] # [N, V, out_dim]
        update_e = self.update_edge(x) # [N, E, out_dim]

        if self.residual:
            update_v = update_v + self.shortcut_node(v)
            update_e = update_e + self.shortcut_edge(e)

        return F.relu(update_v), F.relu(update_e)


class GNN(nn.Module):
    def __init__(self, d_emb, n_layers, node_dim, edge_dim, dropout):
        super(GNN, self).__init__()
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.dropout = dropout

        self.GNN_layers = nn.ModuleList([GNNLayer(node_dim, edge_dim, d_emb)] + \
                                        [GNNLayer(d_emb, d_emb, d_emb) for _ in range(n_layers - 1)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, e, G, A):
        """
        Input:
        - v: (N, V, node_dim)
        - e: (N, E, edge_dim)
        - G: (E, 2)
        - A: (V, V)

        Output:
        - v: (N, V, d_emb)
        - e: (N, E, d_emb)
        """
        for net in self.GNN_layers:
            v, e = net(v, e, G, A)
            v = self.dropout(v)
            e = self.dropout(e)
        return v, e


class PositionalEncoding(nn.Module):
    # batch_first=True
    def __init__(self, d_emb, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_emb, 2, dtype=torch.float) * (-log(10000.0) / d_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_emb)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
