import yaml
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, testState, testGate, configPath='./problems/default_config.yaml', verbose=2):
        self.testState = testState
        self.testGate = testGate
        with open(configPath) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        self.StateOptimizer = BayesianOptimization(
            f=testState,
            pbounds=self.config['pbounds'],
            verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            cost=self.config['cost']['state'],
            random_state=1,
        )

        self.TransferOptimizer = TargetBayesianOptimization(
            f=testGate,
            pbounds=self.config['pbounds'],
            verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            source_bo=self.StateOptimizer,
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

    def default_opt(self, kappa=1, kappa_decay=0.5, kappa_decay_delay=5, acq='ucb', multiAcq='multi_ucb', stateKappa=10):
        pbounds = self.config['pbounds']
        iters = self.config['iters']
        params = pbounds.keys()
        initPoints = []
        for p in params:
            initPoints.append(np.random.uniform(low=pbounds[p][0], high=pbounds[p][1], size=self.config['iters']['init']))

        for i in range(len(initPoints[0])):
            newPoint = {}
            j = 0
            for p in params:
                newPoint[p] = initPoints[j][i]
                j += 1
            stateTarget = self.testState(**newPoint)
            opTarget = self.testGate(**newPoint)
            self.StateOptimizer.register(newPoint, stateTarget)
            self.ControlOptimizer.register(newPoint, opTarget)

        self.StateOptimizer.maximize(
            init_points=0,
            n_iter=iters['state-opt'],
            kappa=stateKappa,
            acq=acq,
        )
        self.TransferOptimizer.transferData(self.StateOptimizer)

        for _ in range(iters['transfer-init']):
            util = UtilityFunction(acq, kappa=1, xi=0.0)
            newPoint = self.StateOptimizer.suggest(util)
            target = self.testGate(**newPoint)
            self.TransferOptimizer.register(newPoint, target)

        self.TransferOptimizer.maximize(
            init_points=0,
            n_iter=iters['transfer-opt'],
        )
        self.TransferOptimizer.maximize(
            init_points=0,
            n_iter=iters['transfer-refine'],
            kappa=0.001
        )

        self.ControlOptimizer.maximize(
            init_points=0,
            n_iter=iters['control-opt'],
            acq=acq,
            kappa=kappa,
            kappa_decay=kappa_decay,
            kappa_decay_delay=kappa_decay_delay,
        )

    def plot_gps(self, stateTitle='State', transferTitle='Gate',
                controlTitle='Control', show=True, save=False, saveFile='./figures/'):

        plot_bo(self.StateOptimizer, stateTitle, save=save, saveDir=saveDir)
        plot_bo(self.TransferOptimizer, transferTitle, save=save, saveDir=saveDir)
        plot_bo(self.ControlOptimizer, controlTitle, save=save, saveDir=saveDir)
        if show:
            plt.show()

    def get_result(self, format_func=fid_to_infidelity):
        transferResult = self.TransferOptimizer.data.bestResult
        transferCosts = self.TransferOptimizer.data.cost
        controlResult = self.ControlOptimizer.data.bestResult
        controlCosts = self.ControlOptimizer.data.cost

        for i in range(len(transferResult)):
            transferResult[i] = format_func(transferResult[i])
        for i in range(len(controlResult)):
            controlResult[i] = format_func(controlResult[i])

        return transferResult, transferCosts, controlResult, controlCosts


    def plot_result(self, title, show=True, save=False, saveFile='./figures/infidelity'):
        iters = self.config['iters']
        cost = self.config['cost']

        transferPoint = (iters['init'] + iters['state-opt']) * cost['state']
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


