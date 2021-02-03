import yaml
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from bayes_opt import BayesianOptimization, TargetBayesianOptimization, UtilityFunction

def plot_bo(bo, title='', save=False, saveFile='./figures/'):
    #ONLY FOR USE WITH 2D PARAMETER SPACES
    #i.e f(x,y)

    X, Y = np.mgrid[0:2:1000j, -2:2:1000j]
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    mean, sigma = bo._gp.predict(positions, return_std=True)
    mean = np.reshape(mean, (-1,1000))
    sigma = np.reshape(sigma, (-1, 1000))
    mean = 1 - mean

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    c1 = ax1.contourf(X,Y, mean, cmap='bwr', levels = 20)
    c2 = ax2.contourf(X,Y, sigma, cmap='bwr', levels = 20)
    plt.colorbar(c1, ax = ax1)
    plt.colorbar(c2, ax = ax2)

    ax1.set_xlabel('a1')
    ax1.set_ylabel('a2')
    ax2.set_xlabel('a1')
    ax2.set_ylabel('a2')

    ax1.set_title(title + ': Mean')
    ax2.set_title(title + ': Sigma')

    if save:
        fig1.savefig(saveFile + title + "_mean.png")
        fig2.savefig(saveFile + title + "_sigma.png")

def fid_to_infidelity(data):
    return 1 - data

class problem:
    def __init__(self, testState_list, testGate, configPath='./problems/default_config.yaml', verbose=2):
        self.testState_list = testState_list
        self.testGate = testGate
        with open(configPath) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        # Primary State Optimizer is the first in the list
        # Data will be transfered from this optimizer
        self.StateOptimizer_list = [BayesianOptimization(
            f=testState,
            pbounds=self.config['pbounds'],
            verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            cost=self.config['cost']['state'],
            random_state=1,
        ) for testState in testState_list]

        self.TransferOptimizer = TargetBayesianOptimization(
            f=testGate,
            pbounds=self.config['pbounds'],
            verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            source_bo_list=self.StateOptimizer_list,
            cost=self.config['cost']['gate'],
            random_state=1,
        )

        self.ControlOptimizer = BayesianOptimization(
            f=testGate,
            pbounds=self.config['pbounds'],
            verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            cost=self.config['cost']['gate'],
            random_state=1,
        )

    def default_opt(self):
        pbounds = self.config['pbounds']
        params = pbounds.keys()
        initPoints = []
        for p in params:
            initPoints.append(np.random.uniform(low=pbounds[p][0], high=pbounds[p][1], size=self.config['state-control-init']))

        for i in range(len(initPoints)):
            newPoint = {}
            j = 0
            for p in params:
                newPoint[p] = initPoints[j][i]
                j += 1
            for k in range(len(self.testState_list)):
                stateTarget = self.testState_list[k](**newPoint)
                self.StateOptimizer_list[k].register(newPoint, stateTarget)
            gateTarget = self.testGate(**newPoint)
            self.ControlOptimizer.register(newPoint, gateTarget)

        for StateOptimizer in self.StateOptimizer_list:
            for optimization in self.config['state'].values():
                if 'refine' in optimization.keys():
                    new_bounds = StateOptimizer.get_new_bounds(optimization['refine'])
                    StateOptimizer.set_bounds(new_bounds)
                StateOptimizer.maximize(
                    init_points=0,
                    n_iter=optimization['iters'],
                    acq=optimization['acq'],
                    kappa=optimization['kappa'],
                    kappa_decay=optimization['kappa-decay'],
                    kappa_decay_delay=optimization['decay-delay'],
                )

        self.TransferOptimizer.transferData(self.StateOptimizer_list[0])
        transferInit = self.config['transfer-init']
        for _ in range(transferInit['iters']):
            stateIndex = np.random.randint(0, len(self.testState_list), 1)[0]
            util = UtilityFunction(transferInit['acq'], kappa=transferInit['kappa'], xi=transferInit['xi'])
            newPoint = self.StateOptimizer_list[stateIndex].suggest(util)
            target = self.testGate(**newPoint)
            self.TransferOptimizer.register(newPoint, target)

        for optimization in self.config['transfer'].values():
            if 'refine' in optimization.keys():
                new_bounds = self.TransferOptimizer.get_new_bounds(optimization['refine'])
                self.TransferOptimizer.set_bounds(new_bounds)
            self.TransferOptimizer.maximize(
                init_points=0,
                n_iter=optimization['iters'],
                acq=optimization['acq'],
                kappa=optimization['kappa'],
                kappa_decay=optimization['kappa-decay'],
                kappa_decay_delay=optimization['decay-delay'],
            )

        for optimization in self.config['control'].values():
            if 'refine' in optimization.keys():
                new_bounds = self.ControlOptimizer.get_new_bounds(optimization['refine'])
                self.ControlOptimizer.set_bounds(new_bounds)
            self.ControlOptimizer.maximize(
                init_points=0,
                n_iter=optimization['iters'],
                acq=optimization['acq'],
                kappa=optimization['kappa'],
                kappa_decay=optimization['kappa-decay'],
                kappa_decay_delay=optimization['decay-delay'],
            )

    def plot_gps(self, stateIndex=0, stateTitle='State', transferTitle='Gate',
                controlTitle='Control', show=True, save=False, saveFile='./figures/'):

        plot_bo(self.StateOptimizer_list[stateIndex], stateTitle, save=save, saveFile=saveFile)
        plot_bo(self.TransferOptimizer, transferTitle, save=save, saveFile=saveFile)
        plot_bo(self.ControlOptimizer, controlTitle, save=save, saveFile=saveFile)
        if show:
            plt.show()

    def get_result(self, format_func=fid_to_infidelity):
        transferResult = deepcopy(self.TransferOptimizer.data.bestResult)
        transferCosts = deepcopy(self.TransferOptimizer.data.cost)
        controlResult = deepcopy(self.ControlOptimizer.data.bestResult)
        controlCosts = deepcopy(self.ControlOptimizer.data.cost)

        for i in range(len(transferResult)):
            transferResult[i] = format_func(transferResult[i])
        for i in range(len(controlResult)):
            controlResult[i] = format_func(controlResult[i])

        return transferResult, transferCosts, controlResult, controlCosts


    def plot_result(self, title='', show=True, save=False, saveFile='./figures/infidelity'):
        cost = self.config['cost']
        totalStateIters = self.config['state-control-init']
        for optimization in self.config['state'].values():
            totalStateIters += optimization['iters']
        transferPoint = totalStateIters * cost['state']

        transferResult, transferCosts, controlResult, controlCosts = self.get_result()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(transferCosts, transferResult, label='With Transfer', color='b')
        ax.plot(controlCosts, controlResult, label='No Transfer', color='r')
        ax.set_xlabel('Cost')
        ax.set_ylabel('Best Infidelity')
        ax.grid('--')
        ax.set_yscale('log')
        ax.axvline(x=transferPoint, color='g', linestyle='--', label='Transfer Point')
        fig.legend()
        if save:
            plt.savefig(saveFile)
        if show:
            plt.show()


