import pickle
import numpy as np
import matplotlib.pyplot as plt


res_nn_pend = pickle.load(open('results/pend_nn.pkl','rb'))
res_lr_pend = pickle.load(open('results/pend_lr.pkl','rb'))
res_nn_acro = pickle.load(open('results/acro_nn.pkl','rb'))
res_lr_acro = pickle.load(open('results/acro_lr.pkl','rb'))
res_nn_mount = pickle.load(open('results/mount_nn.pkl','rb'))
res_lr_mount = pickle.load(open('results/mount_lr.pkl','rb'))

with plt.style.context(['science'], ['ieee']):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    plt.rc('legend', fontsize=16)    # legend fontsize
    axes = axes.flatten()

    # Pendulum
    axes[0].plot(np.median(res_nn_pend, axis=0), label='NN-TRPO - 674 params.')
    axes[0].fill_between(
        np.arange(len(res_nn_pend[0])),
        np.percentile(res_nn_pend, 25, axis=0),
        np.percentile(res_nn_pend, 75, axis=0), color='b', alpha=0.2)

    axes[0].plot(np.median(res_lr_pend, axis=0), label='TRLRPO - 256 params.')
    axes[0].fill_between(
        np.arange(len(res_nn_pend[0])),
        np.percentile(res_lr_pend, 25, axis=0),
        np.percentile(res_lr_pend, 75, axis=0), color='orange', alpha=0.2)
    
    axes[0].set_ylabel('(a) Return', fontsize=18)
    axes[0].set_xlabel('Episodes', fontsize=18)
    axes[0].set_xlim(0, len(res_nn_pend[0]))
    axes[0].set_xticks([0, 500, 1000, 1500, 2000])
    axes[0].set_yticks([-8000, -6000, -4000, -2000, 0])
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].grid()
    axes[0].legend(loc='lower right')

    # Acrobot
    axes[1].plot(np.median(res_nn_acro, axis=0), label='NN-TRPO - 194 params.')
    axes[1].fill_between(
        np.arange(len(res_nn_acro[0])),
        np.percentile(res_nn_acro, 25, axis=0),
        np.percentile(res_nn_acro, 75, axis=0), color='b', alpha=0.2)

    axes[1].plot(np.median(res_lr_acro, axis=0), label='TRLRPO - 32 params.')
    axes[1].fill_between(
        np.arange(len(res_nn_acro[0])),
        np.percentile(res_lr_acro, 25, axis=0),
        np.percentile(res_lr_acro, 75, axis=0), color='orange', alpha=0.2)

    axes[1].set_ylabel('(b) Return', fontsize=18)
    axes[1].set_xlabel('Episodes', fontsize=18)
    axes[1].set_xlim(0, 10000)
    axes[1].set_ylim(-1000, 50)
    axes[1].set_xticks([0, 2500, 5000, 7500, 10000])
    axes[1].set_yticks([-1000, -750, -500, -250, 0])
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].grid()
    axes[1].legend(loc='lower right')

    # Mountain Car
    axes[2].plot(np.median(res_nn_mount, axis=0), label='NN-TRPO - 66 params.')
    axes[2].fill_between(
        np.arange(len(res_nn_mount[0])),
        np.percentile(res_nn_mount, 25, axis=0),
        np.percentile(res_nn_mount, 75, axis=0), color='b', alpha=0.2)

    axes[2].plot(np.median(res_lr_mount, axis=0), label='TRLRPO - 16 params.')
    axes[2].fill_between(
        np.arange(len(res_nn_mount[0])),
        np.percentile(res_lr_mount, 25, axis=0),
        np.percentile(res_lr_mount, 75, axis=0), color='orange', alpha=0.2)

    axes[2].set_ylabel('(c) Return', fontsize=18)
    axes[2].set_xlabel('Episodes', fontsize=18)
    axes[2].set_xlim(0, len(res_nn_mount[0]))
    axes[2].set_ylim(-700, 100)
    axes[2].set_xticks([0, 75, 150, 225, 300])
    axes[2].set_yticks([-600, -450, -300, -150, 0])
    axes[2].tick_params(axis='both', which='major', labelsize=14)
    axes[2].grid()
    axes[2].legend(loc='lower right')

    plt.tight_layout()
    fig.savefig('figures/res.png', dpi=300)
