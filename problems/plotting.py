import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

def plot_bo_1D(bo, label='Default Label', color='r'):
    x = np.linspace(0, 3*np.pi, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1, color=color)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c=color, s=30, zorder=10)

def plot_bo_2D(bo, title='', save=False, saveFile='./figures/'):
    #ONLY FOR USE WITH 2D PARAMETER SPACES
    #i.e f(x,y)

    X, Y = np.mgrid[-1:2.5:1000j, -1.7:1.7:1000j]
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    mean, sigma = bo._gp.predict(positions, return_std=True)
    mean = np.reshape(mean, (-1,1000))
    sigma = np.reshape(sigma, (-1, 1000))
    plt.rcParams.update({'font.size': 18})

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    c1 = ax1.contourf(X/np.pi,Y/np.pi, mean, cmap='bwr', levels = 20)
    c2 = ax2.contourf(X/np.pi,Y/np.pi, sigma, cmap='bwr', levels = 20)
    ax1.set_xlim(-1/4, 3/4)
    ax1.set_ylim(-1/2, 1/2)
    ax1.set_xticks([-1/4, 0, 1/4, 1/2, 3/4])
    ax1.set_xticklabels(['-$\pi$/4', '0', '$\pi$/4', '$\pi$/2', '$3\pi$/4'])
    ax1.set_yticks([-1/2, -1/4, 0, 1/4, 1/2])
    ax1.set_yticklabels(['-$\pi$/2', '-$\pi$/4', '0', '$\pi$/4', '$\pi$/2'])

    ax2.set_xlim(-1/4, 3/4)
    ax2.set_ylim(-1/2, 1/2)
    ax2.set_xticks([-1/4, 0, 1/4, 1/2, 3/4])
    ax2.set_xticklabels(['-$\pi$/4', '0', '$\pi$/4', '$\pi$/2', '$3\pi$/4'])
    ax2.set_yticks([-1/2, -1/4, 0, 1/4, 1/2])
    ax2.set_yticklabels(['-$\pi$/2', '-$\pi$/4', '0', '$\pi$/4', '$\pi$/2'])

    plt.colorbar(c1, ax = ax1)
    plt.colorbar(c2, ax = ax2)

    ax1.set_xlabel('$\phi_1$')
    ax1.set_ylabel('$\phi_2$')
    ax2.set_xlabel('$\phi_1$')
    ax2.set_ylabel('$\phi_2$')

    ax1.set_title(title + ': Mean')
    ax2.set_title(title + ': Uncertainty')

    if save:
        fig1.savefig(saveFile + title + "_mean.png", bbox_inches='tight', pad_inches=0.1)
        fig2.savefig(saveFile + title + "_sigma.png", bbox_inches='tight', pad_inches=0.1)
