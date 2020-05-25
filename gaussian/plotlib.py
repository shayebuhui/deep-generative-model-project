"""Script to plot figures for GAN, adaptted from https://github.com/facebookresearch/GAN-optimization-landscape/blob/master/mnist_plots.ipynb"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

## plt setting  
# plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern,bm}"]
params = {'font.size': 12}
plt.rcParams.update(params)
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def _plot_avg_cosine_similarity(alpha_list, cosine_list_all, norm_list_all, save_path, fig_name, delta=.1):
    
    cos_sim_all = np.stack(cosine_list_all, 1)
    grad_norm_all = np.stack(norm_list_all, 1)
    cos_sim_mu = np.percentile(cos_sim_all, 50, axis=1)
    cos_sim_above = np.percentile(cos_sim_all, 75, axis=1)
    cos_sim_below = np.percentile(cos_sim_all, 25, axis=1)
    grad_norm_mu = np.percentile(grad_norm_all, 50, axis=1)
    grad_norm_above = np.percentile(grad_norm_all, 75, axis=1)
    grad_norm_below = np.percentile(grad_norm_all, 25, axis=1)
    
    fig, ax2 = plt.subplots()
    ax2.plot(alpha_list, grad_norm_mu, c=tableau20[2])
    ax2.set_yscale('log')
    ax2.fill_between(alpha_list, grad_norm_above, grad_norm_below, facecolor=tableau20[2], alpha=0.2)

    ax2.set_ylabel(r'Gradient Norm', fontsize=16)

    ax = ax2.twinx()
    ax.plot(alpha_list, cos_sim_mu)
    ax2.set_xlabel(r'Linear Path', fontsize=16)
    ax.set_ylabel(r'Path Angle', fontsize=16)
    ax2.yaxis.label.set_color(tableau20[2])
    ax2.tick_params(axis='y', colors=tableau20[2])
    ax.yaxis.label.set_color(tableau20[0])
    ax.tick_params(axis='y', colors=tableau20[0])
    ax.fill_between(alpha_list, cos_sim_above, cos_sim_below, facecolor=tableau20[0], alpha=0.2)

    min_cos = np.min(cos_sim_all)
    max_cos = np.max(cos_sim_all)

    ax.set_ylim([(1. + delta) * min_cos - delta * max_cos, (1. + delta) * max_cos - delta * min_cos])
    ax2.grid(True, color="#93a1a1", alpha=0.3)
    ax2.axvline(x=1, color='black', linestyle='--', alpha=0.7)
    ax.axhline(color=tableau20[0], linestyle='--', alpha=0.7)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig('%s/%s.jpg' % (save_path, fig_name), format='jpg', bbox_inches="tight")
    plt.show()

def _plot_eigenvalues(dict_eigs1, dict_eigs2, labels, save_path, fig_name):

    fig1 = plt.figure()
    end = plt.scatter(dict_eigs2['game_eigs'].real, dict_eigs2['game_eigs'].imag, 
                c=[tableau20[2]], label=labels[1], alpha=0.6, edgecolors="black", s=40)
    init = plt.scatter(dict_eigs1['game_eigs'].real, dict_eigs1['game_eigs'].imag, 
                 c=[tableau20[0]], label=labels[0], alpha=0.6, edgecolors="black", s=40)
    plt.grid(True, color="#93a1a1", alpha=0.3)

    os.makedirs(save_path, exist_ok=True)
    
    plt.xlabel('Real Part', fontsize=16)
    plt.ylabel('Imaginary Part', fontsize=16)
    plt.legend()
    plt.savefig('%s/%s-game.jpg' % (save_path, fig_name), format='jpg', bbox_inches="tight")
    plt.show()
    
    fig2 = plt.figure()
    fig2.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(np.arange(0, len(dict_eigs1['gen_eigs']), 2))
    plt.xticks(list(plt.xticks()[0]) + [20])

    for i, eigs in enumerate([dict_eigs1['gen_eigs'], dict_eigs2['gen_eigs']]):
        plt.bar(np.arange(len(eigs)), eigs[::-1], label=labels[i], alpha=0.8)
    plt.xlabel('Top-20 Eigenvalues', fontsize=16)
    plt.ylabel('Magnitude', fontsize=16)
    plt.legend()
    plt.savefig('%s/%s-gen.jpg' % (save_path, fig_name), format='jpg', bbox_inches="tight")
    plt.show()
    
    fig3 = plt.figure()
    fig3.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(np.arange(0, len(dict_eigs1['dis_eigs']+1), 2))
    plt.xticks(list(plt.xticks()[0]) + [20])

    for i, eigs in enumerate([dict_eigs1['dis_eigs'], dict_eigs2['dis_eigs']]):
        plt.bar(np.arange(len(eigs)), eigs[::-1], label=labels[i], alpha=0.8)
    plt.xlabel('Top-20 Eigenvalues', fontsize=16)
    plt.ylabel('Magnitude', fontsize=16)

    plt.legend()
    plt.savefig('%s/%s-dis.jpg' % (save_path, fig_name), format='jpg', bbox_inches="tight")
    plt.show()