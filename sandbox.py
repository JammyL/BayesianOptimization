from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rc('font', family = 'serif', serif = 'cmr10') 
plt.rcParams['mathtext.fontset'] = 'cm' #fonts set such as to be similar to the one in the document
plt.rcParams['axes.titlesize'] = 22

plt.rc('xtick', labelsize=18) 
plt.rc('ytick', labelsize=18) 

def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

x = np.linspace(0, 10, 10000).reshape(-1, 1)
y = target(x)

optimizer = BayesianOptimization(target, {'x': (0, 10)}, random_state=27)

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(12, 9))
    steps = len(optimizer.space)
    """fig.suptitle(
        'n =  {} Steps'.format(steps),
        fontdict={'size':34}
    )"""
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    axis.title.set_text(r'$n = {}$'.format(steps))
    
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=6, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.4, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((0, 10))
    axis.set_ylim((0, 1.5))
    axis.set_ylabel(r'$f(x)$', fontdict={'size':20})
    #axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Acquisition Function', color='red')
    acq.plot(x[np.argmax(utility)], np.max(utility), 'o', markersize=8, 
             label=u'Next Best Guess', markerfacecolor='y', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Acq. func.', fontdict={'size':20})
    acq.set_xlabel(r'$x$', fontdict={'size':20})

    axis.legend(loc=0, fontsize=18)
    acq.legend(loc=0, fontsize=18)
    
    #axis.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.savefig('gp_step_z{}'.format(steps))

optimizer.maximize(init_points=3, n_iter=3, kappa=4)
plot_gp(optimizer, x, y)

optimizer.maximize(init_points=0, n_iter=1, kappa=4)
plot_gp(optimizer, x, y)

optimizer.maximize(init_points=0, n_iter=1, kappa=4)
plot_gp(optimizer, x, y)

optimizer.maximize(init_points=0, n_iter=1, kappa=4)
plot_gp(optimizer, x, y)

optimizer.maximize(init_points=0, n_iter=1, kappa=4)
plot_gp(optimizer, x, y)