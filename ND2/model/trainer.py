import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List
from torch.nn.utils.rnn import pad_sequence
from .model import NDformer
from ..GDExpr import GDExpr

logger = logging.getLogger('ND2.Trainer')


class LossRecord:
    def __init__(self, sample_num:int=0, value_loss:float=0.0, policy_loss:float=0.0, 
                 index_loss:float=0.0):
            self.sample_num = sample_num
            self.value_loss = value_loss
            self.policy_loss = policy_loss
            self.index_loss = index_loss
            self.total_loss = 0.0
            self.logprob = []
            self.top1_acc = []
            self.top5_acc = []
    
    def item(self):
        record = LossRecord(sample_num=self.sample_num,
                            value_loss=self.value_loss.item() if isinstance(self.value_loss, torch.Tensor) else self.value_loss,
                            policy_loss=self.policy_loss.item() if isinstance(self.policy_loss, torch.Tensor) else self.policy_loss,
                            index_loss=self.index_loss.item() if isinstance(self.index_loss, torch.Tensor) else self.index_loss)
        record.total_loss = self.total_loss.item() if isinstance(self.total_loss, torch.Tensor) else self.total_loss
        return record

    def set_total_loss(self, w_value=1.0, w_policy=1.0, w_index=1.0):

        self.total_loss = w_value * self.value_loss + \
                          w_policy * self.policy_loss + \
                          w_index * self.index_loss
    @staticmethod
    def _weighted_update(value, weight, new_value, new_weight):
        return (value * weight + new_value * new_weight) / (weight + new_weight)

    def update(self, other:'LossRecord'):
        if other is None: return
        self.value_loss = self._weighted_update(self.value_loss, self.sample_num,
                                                other.value_loss, other.sample_num)
        self.policy_loss = self._weighted_update(self.policy_loss, self.sample_num, 
                                                other.policy_loss, other.sample_num)
        self.index_loss = self._weighted_update(self.index_loss, self.sample_num, 
                                               other.index_loss, other.sample_num)
        self.total_loss = self._weighted_update(self.total_loss, self.sample_num,
                                               other.total_loss, other.sample_num)
        self.sample_num += other.sample_num
        self.logprob.extend(other.logprob)
        self.top1_acc.extend(other.top1_acc)
        self.top5_acc.extend(other.top5_acc)

    def __repr__(self):
        return f'LossRecord(total_loss={self.total_loss:7.4f}, ' + \
               f'policy_loss={self.policy_loss:7.4f}, ' + \
               f'index_loss={self.index_loss:7.4f}' + \
               f'value_loss={self.value_loss:7.4f})'


class Trainer(object):
    def __init__(self, model:NDformer,
                 device='cpu',
                 continue_from=None,
                 label_smoothing=0.1,
                 clip_grad_norm=100.0,
                 learning_rate=1.0,
                 warmup_steps=4000,
                 min_token_count=1000,
                 ):
        super().__init__()
        self.model = model
        self.device = device
        self.continue_from = continue_from
        self.label_smoothing = label_smoothing
        self.max_clip_grad_norm = clip_grad_norm
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.min_token_count = min_token_count
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: self.model.decoder.d_emb ** -0.5 * \
                min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5))
        self.value_ctr = nn.MSELoss()
        self.policy_ctr = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.index_ctr = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.batch = None
        self.token_count = 0
        self.sample_count = 0
        self.train_step_count = 0
        self.log_record = {'train': [], 'valid': [], 'test': []}

        logger.info(self.get_model_info())
        if continue_from is not None:
            self.load_checkpoint(continue_from)

    def get_model_info(self):
        enc_params = sum([p.numel() for p in self.model.encoder.parameters() if p.requires_grad])
        enc_GNN_params = sum([p.numel() for p in self.model.encoder.GNN.parameters() if p.requires_grad])
        enc_transformer_params = sum([p.numel() for p in self.model.encoder.Transformer.parameters() if p.requires_grad])
        dec_params = sum([p.numel() for p in self.model.decoder.parameters() if p.requires_grad])
        dec_embedding_params = sum([p.numel() for p in self.model.decoder.token_embedding.parameters() if p.requires_grad])
        dec_index_embedding_params = sum([p.numel() for p in self.model.decoder.index_embedding.parameters() if p.requires_grad])
        dec_transformer_params = sum([p.numel() for p in self.model.decoder.decoder.parameters() if p.requires_grad])
        return f'#Trainable Parameters: {enc_params+dec_params:10,}\n' + \
               f'  - Encoder:           {enc_params:10,}\n' + \
               f'      - GNN:           {enc_GNN_params:10,}\n' + \
               f'      - Transformer:   {enc_transformer_params:10,}\n' + \
               f'  - Decoder:           {dec_params:10,}\n' + \
               f'      - Embedding:     {dec_embedding_params:10,}\n' + \
               f'      - Idx-Embedding: {dec_index_embedding_params:10,}\n' + \
               f'      - Transformer:   {dec_transformer_params:10,}\n'
    
    def step(self, var_dict:dict,
                   root_type:str, 
                   prefixes:List[torch.LongTensor],
                   values:torch.FloatTensor,
                   policies:torch.LongTensor,
                   indexes:torch.LongTensor,
                   parents:List[torch.LongTensor], 
                   types:List[torch.LongTensor],
                   force=False, train=True, detail=False) -> LossRecord:
        """
        Arguments:
        - var_dict: Dict[A, G, out, variables...]
        - root_type: str
        - prefixes: List[torch.LongTensor(L)], len=M
            start with <SOS>, end with <EOS>
        - values: None (or torch.FloatTensor(M))
        - policies: torch.LongTensor(M) (or torch.LongTensor(M, n_words))
        - indexes: torch.LongTensor(M) (or torch.LongTensor(M, max_seq_len))
        - parents: List[torch.LongTensor(L)], len=M
        - types: List[torch.LongTensor(L)], len=M
        - force: bool, whether to force to update the model
        - train: bool, whether to update the parameters
        """
        if train: self.model.train()
        else: self.model.eval()
        with torch.set_grad_enabled(mode=train):
            if self.batch is None:
                self.batch = dict(data_emb=[], prefixes=[], policies=[],
                                indexes=[], parents=[], types=[], values=[])
                self.token_count = 0
                self.sample_count = 0
            data_emb = self.model.encode(root_type, var_dict) # (N, d_emb)
            self.batch['data_emb'].append(data_emb.unsqueeze(0).expand(len(prefixes), -1, -1)) # (m, N, d_emb)
            self.batch['prefixes'].extend(prefixes) # (m, <=L)
            self.batch['policies'].append(policies) # (m,) or (m, n_words)
            self.batch['indexes'].append(indexes) # (m,) or (m, max_seq_len)
            self.batch['parents'].extend(parents) # (m, <=L)
            self.batch['types'].extend(types) # (m, <=L)
            self.token_count += sum([prefix.numel() for prefix in prefixes])
            self.sample_count += len(prefixes)
            
            if self.token_count < self.min_token_count and not force: return None

            N = max(data_emb.shape[1] for data_emb in self.batch['data_emb'])
            for i in range(len(self.batch['data_emb'])):
                data_emb = self.batch['data_emb'][i]
                L = data_emb.shape[1]
                self.batch['data_emb'][i] = F.pad(data_emb, (0, 0, 0, N - L), value=GDExpr.pad_id)
            data_emb = torch.cat(self.batch['data_emb'], dim=0) # (M, N, d_emb)
            prefixes = pad_sequence(self.batch['prefixes'], batch_first=True, 
                                    padding_value=GDExpr.pad_id) # (M, L)
            parents = pad_sequence(self.batch['parents'], batch_first=True, 
                                   padding_value=GDExpr.pad_id) # (M, L)
            types = pad_sequence(self.batch['types'], batch_first=True, 
                                 padding_value=GDExpr.pad_id) # (M, L)
            gt_policies = torch.cat(self.batch['policies'], dim=0) # (M, n_words)
            gt_indexes = torch.cat(self.batch['indexes'], dim=0) # (M, max_seq_len)
            self.clear_batch()

            prefixes = prefixes.to(self.device)
            gt_policies = gt_policies.to(self.device)
            gt_indexes = gt_indexes.to(self.device)
            pd_values, pd_policies, pd_indexes = self.model.decode(data_emb, prefixes, parents, types)

            loss_record = LossRecord(sample_num=gt_policies.numel(),
                                    policy_loss=self.policy_ctr(pd_policies, gt_policies),
                                    index_loss=self.index_ctr(pd_indexes, gt_indexes))
            loss_record.set_total_loss(w_policy=1.0, 
                                       w_index=1.0, 
                                       w_value=1.0)

            if train:
                self.optimizer.zero_grad()
                loss_record.total_loss.backward()
                self.clip_grad_norm()
                self.optimizer.step()
                self.scheduler.step()
                self.train_step_count += 1
            if detail:
                index = gt_policies.unsqueeze(-1)
                loss_record.logprob = pd_policies.log_softmax(dim=-1) \
                                                 .gather(dim=-1, index=index) \
                                                 .squeeze(-1).tolist()
                loss_record.top1_acc = pd_policies.topk(k=1, dim=-1)[1] \
                                                  .eq(index).any(dim=-1).tolist()
                loss_record.top5_acc = pd_policies.topk(k=5, dim=-1)[1] \
                                                  .eq(index).any(dim=-1).tolist()
            return loss_record

    def load_checkpoint(self, path:str, load_model_only=False):
        if load_model_only:
            self.model.load(path)
            logger.info(f'Load model from {path}')
        else:
            self.model.load(path, self.optimizer, self.scheduler)
            logger.info(f'Load model, optimizer and scheduler from {path}')
    
    def save_checkpoint(self, path:str, save_model_only=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if save_model_only:
            self.model.save(path)
            logger.info(f'Save model to {path}')
        else:
            self.model.save(path, self.optimizer, self.scheduler)
            logger.info(f'Save model, optimizer and scheduler to {path}')

    def clear_batch(self):
        self.batch = None
        self.token_count = 0
        self.sample_count = 0

    def record(self, epoch:int, step:int, loss_record:LossRecord, mode:str):
        if mode == 'train':
            self.log_record['train'].append((epoch, step, loss_record.item()))
        elif mode == 'valid':
            self.log_record['valid'].append((epoch, step, loss_record.item()))
        elif mode == 'test':
            self.log_record['test'].append((epoch, step, loss_record.item(), 
                                            np.mean(loss_record.logprob),
                                            np.mean(loss_record.top1_acc),
                                            np.mean(loss_record.top5_acc)))

    def clip_grad_norm(self):
        get_max_norm = lambda module: max([p.grad.norm(p=2).item() 
                                           for p in module.parameters() 
                                           if p.requires_grad and p.grad is not None])
        max_norm = get_max_norm(self.model)
        if max_norm > 1e6 or np.isnan(max_norm):
            max_norm_detail = {'encoder-GNN': get_max_norm(self.model.encoder.GNN),
                               'encoder-Transformer': get_max_norm(self.model.encoder.Transformer),
                               'decoder-embedding': get_max_norm(self.model.decoder.token_embedding),
                               'decoder-index-embedding': get_max_norm(self.model.decoder.index_embedding),
                               'decoder-Transformer': get_max_norm(self.model.decoder.decoder)}
            logger.warning(f'Gradient norm is {max_norm:.4f}\n'
                           '\n'.join([f'* {k}: {v:.4f}' for k, v in max_norm_detail.items()]))
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
        logger.debug(f'Clip grad norm: {max_norm:.4f} -> {get_max_norm(self.model):.4f}')

    def plot(self, path:str):        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.rcParams['font.size'] = 7
        fig = plt.figure(figsize=(9, 12), dpi=300)
        grid = gridspec.GridSpec(4, 3, figure=fig)
        axes = [
            fig.add_subplot(grid[0, :]), # Total loss
            fig.add_subplot(grid[2, :]), # Detailed loss
            fig.add_subplot(grid[1, 0]), # Policy Loss
            fig.add_subplot(grid[1, 1]), # Index Loss
            fig.add_subplot(grid[1, 2]), # Value Loss
            fig.add_subplot(grid[3, :]), # Logprob, Top1, Top5
        ]
        
        train_epochs, train_steps, train_records = zip(*self.log_record['train'])
        _, valid_steps, valid_records = zip(*self.log_record['valid'])
        _, test_steps, test_records, logprobs, top1_accs, top5_accs = zip(*self.log_record['test'])

        diff = np.diff(train_epochs, prepend=-1)
        for i in np.nonzero(diff)[0]:
            axes[0].axvline(train_steps[i], color='k', linestyle='--', linewidth=0.5 * diff[i])
            axes[1].axvline(train_steps[i], color='k', linestyle='--', linewidth=0.5 * diff[i])

        def plot_curve(ax, x, y, sample_num=100, label_prefix='Final', **kwargs):
            _x = np.array_split(x, min(sample_num, len(x)))
            _y = np.array_split(y, min(sample_num, len(y)))
            kwargs.pop('label', None)
            ax.plot(x, y, **kwargs, alpha=0.3)
            ax.plot([np.mean(i) for i in _x], [np.mean(i) for i in _y], **kwargs, label=f'{label_prefix}={_y[-1].mean():.4f}')

        plot_curve(axes[0], train_steps, [r.total_loss for r in train_records], label_prefix=f'Train', color='C0')
        plot_curve(axes[0], valid_steps, [r.total_loss for r in valid_records], label_prefix=f'Valid', color='C0', linestyle='--')
        plot_curve(axes[0], test_steps, [r.total_loss for r in test_records], label_prefix=f'Test', color='C0', linestyle=':')
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('Total Loss')
        axes[0].semilogy()
        axes[0].grid('on', zorder=-1, alpha=0.2, dashes=[3, 1])
        axes[0].legend(ncol=3)

        plot_curve(axes[1], train_steps, [r.policy_loss for r in train_records], color='C1')
        plot_curve(axes[1], valid_steps, [r.policy_loss for r in valid_records], color='C1', linestyle='--')
        plot_curve(axes[1], test_steps, [r.policy_loss for r in test_records], color='C1', linestyle=':')
        plot_curve(axes[1], train_steps, [r.index_loss for r in train_records], color='C2')
        plot_curve(axes[1], valid_steps, [r.index_loss for r in valid_records], color='C2', linestyle='--')
        plot_curve(axes[1], test_steps, [r.index_loss for r in test_records], color='C2', linestyle=':')
        plot_curve(axes[1], train_steps, [r.value_loss for r in train_records], color='C3')
        plot_curve(axes[1], valid_steps, [r.value_loss for r in valid_records], color='C3', linestyle='--')
        plot_curve(axes[1], test_steps, [r.value_loss for r in test_records], color='C3', linestyle=':')
        axes[1].set_xlabel('step')
        axes[1].set_ylabel('Detailed Loss')
        axes[1].semilogy()
        axes[1].grid('on', zorder=-1, alpha=0.2, dashes=[3, 1])
        # axes[1].legend(ncol=5)

        plot_curve(axes[2], train_steps, [r.policy_loss for r in train_records], label_prefix=f'Train', color='C1')
        plot_curve(axes[2], valid_steps, [r.policy_loss for r in valid_records], label_prefix=f'Valid', color='C1', linestyle='--')
        plot_curve(axes[2], test_steps, [r.policy_loss for r in test_records], label_prefix=f'Test', color='C1', linestyle=':')
        axes[2].legend()
        axes[2].set_ylabel('Policy Loss')
        axes[2].semilogy()
        axes[2].grid('on', zorder=-1, alpha=0.2, dashes=[3, 1])

        plot_curve(axes[3], train_steps, [r.index_loss for r in train_records], label_prefix=f'Train', color='C2')
        plot_curve(axes[3], valid_steps, [r.index_loss for r in valid_records], label_prefix=f'Valid', color='C2', linestyle='--')
        plot_curve(axes[3], test_steps, [r.index_loss for r in test_records], label_prefix=f'Test', color='C2', linestyle=':')
        axes[3].legend()
        axes[3].set_ylabel('Index Loss')
        axes[3].semilogy()
        axes[3].grid('on', zorder=-1, alpha=0.2, dashes=[3, 1])

        plot_curve(axes[4], train_steps, [r.value_loss for r in train_records], label_prefix=f'Train', color='C3')
        plot_curve(axes[4], valid_steps, [r.value_loss for r in valid_records], label_prefix=f'Valid', color='C3', linestyle='--')
        plot_curve(axes[4], test_steps, [r.value_loss for r in test_records], label_prefix=f'Test', color='C3', linestyle=':')
        axes[4].legend()
        axes[4].set_ylabel('Value Loss')
        axes[4].semilogy()
        axes[4].grid('on', zorder=-1, alpha=0.2, dashes=[3, 1])

        plot_curve(axes[5], test_steps, np.exp(logprobs), label_prefix='SS-Prob', color='C0')
        plot_curve(axes[5], test_steps, top1_accs, label_prefix='top1-acc', color='C1')
        plot_curve(axes[5], test_steps, top5_accs, label_prefix='top5-acc', color='C2')
        axes[5].set_xlabel('step')
        axes[5].legend(ncol=3)
        axes[5].set_ylabel('Test Result')
        axes[5].set_ylim(0, 1)
        axes[5].grid('on', zorder=-1, alpha=0.2, dashes=[3, 1])

        fig.tight_layout()
        fig.savefig(path)
        plt.close()
        logger.info(f'Plot loss curves to {path}')