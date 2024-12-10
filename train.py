import time
import signal
import logging
import setproctitle
import numpy as np
import torch.utils.data as D
from argparse import ArgumentParser
from ND2.dataset import Dataset
from ND2.model import NDformer
from ND2.utils import AutoGPU, init_logger, seed_all
from ND2.model.trainer import Trainer, LossRecord
from ND2.GDExpr import GDExpr

logger = logging.getLogger('ND2.train')
def handler(signum, frame): raise KeyboardInterrupt    
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

def main(args):
    seed_all(args.seed)
    init_logger(args.name, f'./log/train/{args.name}/info.log', root_name='ND2')
    setproctitle.setproctitle(f'{args.name}@ZihanYu')

    if args.device == 'auto':
        args.device = AutoGPU().choice_gpu(10*1024, interval=15)

    logger.info(f'args: {args}')
    logger.info(f'GDExpr.config: {GDExpr.config}')

    # %% Load data
    dataset = Dataset.load_csv([
        './data/pretrain/node-easy.csv',
        './data/pretrain/node-middle.csv',
        './data/pretrain/node-hard.csv',
        './data/pretrain/node-polinomial.csv',
        './data/pretrain/edge-easy.csv',
        './data/pretrain/edge-middle.csv',
        './data/pretrain/edge-hard.csv',
        './data/pretrain/edge-polinomial.csv',
        './data/pretrain/node-GD.csv',
    ], n_expr_sample=args.n_expr_sample, device=args.device)
    train_set, valid_set = D.random_split(dataset, [len(dataset) - args.n_valid_set, args.n_valid_set])
    test_set = Dataset.load_csv([
        './data/synthetic/synthetic.csv'
    ], n_expr_sample=args.n_expr_sample, device=args.device)

    # %% Load model
    model = NDformer(device=args.device)
    trainer = Trainer(model, device=args.device)
    if args.continue_from is not None:
        trainer.load_checkpoint(args.continue_from)

    # %% Train
    for epoch in range(args.n_epochs):
        for train_idx, sample_idx in enumerate(np.random.permutation(len(train_set))):
            model.train()
            loss = trainer.step(*train_set[sample_idx], force=False, train=True)
            if loss is None: continue
            trainer.record(epoch, trainer.train_step_count, loss, 'train')
            log = {
                'Step': f'{trainer.train_step_count}',
                'Epoch': f'{epoch}',
                'Train-idx': f'{train_idx}',
                'Train-loss': f'{loss.total_loss:.4f}',
                'Policy-Loss': f'{loss.policy_loss:.4f}',
                'Index-Loss': f'{loss.index_loss:.4f}',
                'LR': f'{trainer.scheduler.get_last_lr()[0]:.4e}'
            }
            logger.info(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))

            if trainer.train_step_count % args.valid_every_step == 0:
                model.eval()
                assert trainer.batch is None # trainer.clear_batch()
                loss_record = LossRecord()
                for valid_idx, sample in enumerate(valid_set):
                    loss = trainer.step(*sample, force=(valid_idx==len(valid_set)-1), train=False)
                    if loss is None: continue
                    loss_record.update(loss)
                    logger.debug(f'Epoch {epoch}, Step {trainer.train_step_count}, Valid-idx: {valid_idx}, '
                                f'Valid loss: {loss.total_loss:.4f} (policy: {loss.policy_loss:.4f}, '
                                f'index: {loss.index_loss:.4f})')
                log = {
                    'Epoch': f'{epoch}',
                    'Step': f'{trainer.train_step_count}',
                    'Valid-loss': f'{loss_record.total_loss:.4f}',
                    'Policy-Loss': f'{loss_record.policy_loss:.4f}',
                    'Index-Loss': f'{loss_record.index_loss:.4f}'
                }
                logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))
                trainer.record(epoch, trainer.train_step_count, loss_record, 'valid')
                assert trainer.batch is None # trainer.clear_batch()
            
            if trainer.train_step_count % args.test_every_step == 0:
                model.eval()
                assert trainer.batch is None # trainer.clear_batch()
                loss_record = LossRecord()
                for test_idx, sample in enumerate(test_set):
                    loss = trainer.step(*sample, force=True, train=False, detail=True)
                    loss_record.update(loss)
                    logger.debug(f'[{test_set.datafiles[test_idx]}] '
                                f'logprob: {np.mean(loss.logprob):.4f} ({np.exp(np.mean(loss.logprob)):.2%}), '
                                f'top1: {np.mean(loss.top1_acc):.2%}, '
                                f'top5: {np.mean(loss.top5_acc):.2%}')
                log = {
                    'Epoch': f'{epoch}',
                    'Step': f'{trainer.train_step_count}',
                    'Test-logprob': f'{np.mean(loss_record.logprob):.4f}',
                    'Test-top1': f'{np.mean(loss_record.top1_acc):.2%}',
                    'Test-top5': f'{np.mean(loss_record.top5_acc):.2%}'
                }
                logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))
                trainer.record(epoch, trainer.train_step_count, loss_record, 'test')
                assert trainer.batch is None # trainer.clear_batch()

            if trainer.train_step_count % args.save_every_step == 0:
                trainer.save_checkpoint(f'./log/train/{args.name}/checkpoint.pth')

            if trainer.train_step_count % args.plot_every_step == 0:
                trainer.plot(f'./log/train/{args.name}/plot.png')

            if len(str(trainer.train_step_count).rstrip('0')) == 1 and trainer.train_step_count > 99:
                trainer.save_checkpoint(f'./log/train/{args.name}/checkpoints/checkpoint-{trainer.train_step_count}.pth', save_model_only=True)
                trainer.plot(f'./log/train/{args.name}/plots/plot-{trainer.train_step_count}.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=None)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-d', '--device', type=str, default='cuda:1')
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--n_expr_sample', type=int, default=10)
    parser.add_argument('--n_valid_set', type=int, default=500)
    parser.add_argument('--plot_every_step', type=int, default=50)
    parser.add_argument('--save_every_step', type=int, default=100)
    parser.add_argument('--test_every_step', type=int, default=10)
    parser.add_argument('--valid_every_step', type=int, default=10)
    parser.add_argument('--continue_from', type=str, default=None)
    args = parser.parse_args()

    args.name = args.name or 'Train_' + time.strftime('%Y%m%d_%H%M%S')
    main(args)
