import os
import sys
import yaml
import json
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from ND2.GDExpr import GDExpr
from ND2.utils import AttrDict
from argparse import ArgumentParser

def plot_first_N(data, ax, N, label, cmap):
    if data[0].ndim == 1:
        data[0] = data[0][:, np.newaxis].repeat(data[1].shape[1], axis=1)
    for i in range(N):
        ax.plot(*[x[..., i] for x in data], label=f'{label} {i+1}', color=cmap(i/N))
    for i in range(N, data[0].shape[-1]):
        ax.plot(*[x[..., i] for x in data], color='#b2bec3', alpha=0.3, zorder=-100000)


def main(args):
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.plot, exist_ok=True)
    for name, config in AttrDict.load_yaml(args.conf).items():
        random.seed(0)
        np.random.seed(0)
        
        # Network Topolocy
        if config.network.get('use', None) is not None:
            A = np.loadtxt(config.network.use, delimiter=',').astype(int)
            V = A.shape[0]
            E = int(np.sum(A))
        else:
            V = config.network.V
            E = config.network.E
            A = np.zeros((V, V))
            if not config.network.get('direction', False):
                assert E % 2 == 0, 'E should be even when direction is False.'
                tmp = np.triu_indices(V, 1)
                idx = np.random.choice(len(tmp[0]), E//2, replace=False)
                A[tmp[0][idx], tmp[1][idx]] = 1
                A[tmp[1][idx], tmp[0][idx]] = 1
            elif config.network.get('DAG', False):
                tmp = np.stack(np.triu_indices(V, +1), axis=-1)
                idx = np.random.choice(tmp.shape[0], E, replace=False)
                A[tmp[idx, 0], tmp[idx, 1]] = 1
            else:
                tmp = np.concatenate([
                    np.stack(np.triu_indices(V, +1), axis=-1),
                    np.stack(np.tril_indices(V, -1), axis=-1)], axis=0)
                idx = np.random.choice(tmp.shape[0], E, replace=False)
                A[tmp[idx, 0], tmp[idx, 1]] = 1
        G = np.stack(np.nonzero(A), axis=-1)
        assert A.shape == (V, V)
        assert G.shape == (E, 2)

        # Pre-process
        GD_exprs = {var: GDExpr.sympy2prefix(GDExpr.parse_expr(cfg.GD_expr), config.root_type, 
                                            reindex=False, keep_coeff=True) 
                    for var, cfg in config.variables.dependent.items()}

        # Simulation
        simulation = {var: [] for var in config.variables.independent | config.variables.dependent}
        for step_idx, t in enumerate(np.arange(config.N) * config.dt):
            for var in config.variables.independent:
                N, dt = config.N, config.dt
                if step_idx == 0:
                    simulation[var].append(eval(config.variables.independent[var].initialize))
                else:
                    for v in simulation: 
                        locals()[v] = simulation[v][-1]
                    if 'update' in config.variables.independent[var]:
                        simulation[var].append(eval(config.variables.independent[var].update))
            for var in config.variables.dependent:
                var_dict = {var:simulation[var][-1] for var in config.variables.independent}
                var_dict = {**var_dict, 'A': A, 'G': G}
                simulation[var].append(GDExpr.eval(GD_exprs[var], var_dict, [], strict=False))
        for var in simulation:
            simulation[var] = np.concatenate(simulation[var], axis=0)
            if simulation[var].shape[0] == 1:
                simulation[var] = simulation[var][0]

        # Save
        var_dict = {'V': V, 'E': E, 'A': A, 'G': G, **simulation}
        for k, v in var_dict.items():
            if isinstance(v, np.ndarray):
                var_dict[k] = v.tolist()
        json.dump(var_dict, open(os.path.join(args.save, f'{name}.json'), 'w'))

        # Plot
        plt.rcParams['font.size'] = 7  # 使用 7pt 无衬线字体。若为衬线体则用 8pt
        RN1, CN1 = int(np.ceil((len(simulation)) / 3)), 3
        FW1 = 17.6 / 2.54  # 8.3 cm，论文单栏宽度。双栏则为 17.6cm
        LM1, RM1, TM1, BM1 = np.array([6, 0.5, 3, 4]) * 7/72  # 7pt 的字体，1pt 对应 1/72 inch
        HS1, VS1 = np.array([6.0, 6.0]) * 7/72  # 一般可取 LM1, BM
        AW1 = (FW1 - RM1 - LM1 - (CN1 - 1) * HS1) / CN1
        AH1 = AW1 / 1.0
        FH1 = RN1 * AH1 + (RN1 - 1) * VS1 + TM1 + BM1

        RN2, CN2 = 1, 3
        FW2 = FW1  # 8.3 cm，论文单栏宽度。双栏则为 17.6cm
        LM2, RM2, TM2, BM2 = np.array([6, 0.5, 3, 4]) * 7/72  # 7pt 的字体，1pt 对应 1/72 inch
        HS2, VS2 = np.array([6.0, 6.0]) * 7/72  # 一般可取 LM2, BM
        AW2 = (FW2 - RM2 - LM2 - (CN2 - 1) * HS2) / CN2
        AH2 = AW2 / 1.0
        FH2 = RN2 * AH2 + (RN2 - 1) * VS2 + TM2 + BM2

        FW, FH = FW1, FH1 + FH2
        LM, RM, TM, BM = LM1, RM1, TM1, BM2
        HS, VS = 0, 0
        fig = plt.figure(figsize=(FW, FH), dpi=300)
        fig.subplots_adjust(left=LM/FW, right=1-RM/FW, top=1-TM/FH, bottom=BM/FH)
        grid = gridspec.GridSpec(RN1+RN2, 1, figure=fig)
        grid1 = gridspec.GridSpecFromSubplotSpec(RN1, CN1, subplot_spec=grid[:RN1, :], wspace=HS1/AW1, hspace=VS1/AH1)
        grid2 = gridspec.GridSpecFromSubplotSpec(RN2, CN2, subplot_spec=grid[RN1:, :], wspace=HS2/AW2, hspace=VS2/AH2)
        axes1 = [fig.add_subplot(grid1[i, j]) for i in range(RN1) for j in range(CN1)]
        axes2 = [fig.add_subplot(grid2[i, j]) for i in range(RN2) for j in range(CN2)]

        from matplotlib.colors import LinearSegmentedColormap
        my_hsv = LinearSegmentedColormap.from_list('my_hsv', ['#d63031', '#e17055', '#fdcb6e', '#00b894', '#00cec9', '#0984e3' ,'#6c5ce7', '#e84393'])
        
        time = np.arange(config.N) * config.dt
        for idx, var in enumerate(config.variables.independent):
            ax = axes1[idx]
            if simulation[var].ndim == 1:
                ax.bar(np.arange(len(simulation[var])), simulation[var], color='#636e72', width=0.75)
                ax.set_xlabel(f'{config.variables.independent[var].type} index')
            else:
                plot_first_N([time, simulation[var]], ax=ax, N=10, label=config.variables.independent[var].type, cmap=my_hsv)
                ax.set_xlabel('time / s')
            ax.set_ylabel(var)
        
        for idx2, var in enumerate(config.variables.dependent):
            ax = axes1[idx+1+idx2]
            plot_first_N([time, simulation[var]], ax=ax, N=10, label=config.variables.dependent[var].type, cmap=my_hsv)
            ax.set_xlabel('time / s')
            ax.set_ylabel(var)
        
        for i in range(idx+1+idx2+1, RN1*CN1): axes1[i].set_visible(False)

        axes2[0].imshow(A, cmap='gray_r')
        # axes2[0].axis('off')
        # for i in range(A.shape[0]):
        #     axes2[0].hlines(-0.49, i-0.5, i+0.5, color=my_hsv(i/A.shape[0]), linewidth=1.0)
        #     axes2[0].vlines(-0.49, i-0.5, i+0.5, color=my_hsv(i/A.shape[0]), linewidth=1.0)
        axes2[0].set_xticks([])
        axes2[0].set_yticks([])
        axes2[0].set_title(f'Network Topology (V={V}, E={E}), $\\bar{{d}}$={2*E/V:.2f})')

        if 'plot' in config:
            ax = axes2[1]
            if len(config.plot) == 1:
                data = eval(config.plot[0], globals(), locals() | simulation | {'sin': np.sin})
                plot_first_N([time, data], ax=ax, N=10, label='', cmap=my_hsv)
            elif len(config.plot) == 2:
                data = [
                    eval(config.plot[0], globals(), locals() | simulation | {'sin': np.sin}),
                    eval(config.plot[1], globals(), locals() | simulation | {'sin': np.sin})
                ]
                plot_first_N(data, ax=ax, N=10, label='', cmap=my_hsv)
            elif len(config.plot) == 3:
                ax.set_visible(False)
                pos= ax.get_position()
                ax = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height], projection='3d')
                ax.view_init(elev=30, azim=30)
                data = [
                    eval(config.plot[0], globals(), locals() | simulation | {'sin': np.sin}),
                    eval(config.plot[1], globals(), locals() | simulation | {'sin': np.sin}),
                    eval(config.plot[2], globals(), locals() | simulation | {'sin': np.sin})
                ]
                plot_first_N(data, ax=ax, N=10, label='', cmap=my_hsv)
            else:
                raise NotImplementedError

        fig.suptitle(name)
        fig.savefig(os.path.join(args.plot, f'{name}.png'))
        plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default='./config/synthetic_LFR.yaml')
    parser.add_argument('--save', type=str, default='./data/synthetic/')
    parser.add_argument('--plot', type=str, default='./plot/synthetic/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)