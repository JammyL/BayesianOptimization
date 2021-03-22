import yaml
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from bayes_opt import BayesianOptimization, TargetBayesianOptimization, UtilityFunction
from .plotting import plot_bo_1D, plot_bo_2D

def fid_to_infidelity(data):
    return 1 - data

def no_change(data):
    return data

class problem:
    def __init__(self, testState_list, testGate, configPath='./problems/default_config.yaml', verbose=2):
        self.testState_list = testState_list
        self.testGate = testGate
        with open(configPath) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        if 'transfer-init' in self.config.keys():
            if not 'type' in self.config['transfer-init'].keys():
                self.config['transfer-init']['type'] = 'state'

        # Primary State Optimizer is the first in the list
        # Data will be transfered from this optimizer
        if 'state' in self.config.keys():
            self.StateOptimizer_list = [BayesianOptimization(
                f=testState,
                pbounds=self.config['pbounds'],
                verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                cost=self.config['cost']['state'],
                random_state=1,
            ) for testState in testState_list]
        else:
            self.StateOptimizer_list = []
        if 'transfer' in self.config.keys() and 'state' in self.config.keys():
            self.TransferOptimizer = TargetBayesianOptimization(
                f=testGate,
                pbounds=self.config['pbounds'],
                verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                source_bo_list=self.StateOptimizer_list,
                cost=self.config['cost']['gate'],
                random_state=1,
            )
        else:
            self.TransferOptimizer = None
        if 'control' in self.config.keys():
            self.ControlOptimizer = BayesianOptimization(
                f=testGate,
                pbounds=self.config['pbounds'],
                verbose=verbose, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                cost=self.config['cost']['gate'],
                random_state=1,
            )
        else:
            self.ControlOptimizer = None
        if 'state-control-init' in self.config.keys():
            self.config['state-init'] = self.config['state-control-init']
            self.config['control-init'] = self.config['state-control-init']

    def default_opt(self):
        pbounds = self.config['pbounds']
        params = pbounds.keys()
        initPoints = []
        stateIters = len(self.StateOptimizer_list) * self.config['state-init']
        controlIters = self.config['control-init']
        for p in params:
            initPoints.append(np.random.uniform(low=pbounds[p][0], high=pbounds[p][1], size=max(stateIters, controlIters)))
        for i in range(max(stateIters, controlIters)):
            newPoint = {}
            j = 0
            for p in params:
                newPoint[p] = initPoints[j][i]
                j += 1
            gateTarget = self.testGate(**newPoint)
            if self.StateOptimizer_list != [] and i < stateIters:
                stateIndex = i % len(self.StateOptimizer_list)
                stateTarget = self.testState_list[stateIndex](**newPoint)
                self.StateOptimizer_list[stateIndex].register(newPoint, stateTarget)

            if self.ControlOptimizer != None and i < controlIters:
                self.ControlOptimizer.register(newPoint, gateTarget)

        for StateOptimizer in self.StateOptimizer_list:
            for optimization in self.config['state'].values():
                if 'refine' in optimization.keys():
                    new_bounds = StateOptimizer.get_new_bounds(optimization['refine'])
                    StateOptimizer.set_bounds(new_bounds)
                if not 'kappa-min' in optimization.keys():
                    optimization['kappa-min'] = 0
                if 'decay-delay' in optimization.keys() and 'kappa-delay' not in optimization.keys():
                    optimization['kappa-delay'] = optimization['decay-delay']
                StateOptimizer.maximize(
                    init_points=0,
                    n_iter=optimization['iters'],
                    acq=optimization['acq'],
                    kappa=optimization['kappa'],
                    kappa_decay=optimization['kappa-decay'],
                    kappa_decay_delay=optimization['kappa-delay'],
                    kappa_min=optimization['kappa-min'],
                )

        if self.TransferOptimizer != None:
            self.TransferOptimizer.transferData(self.StateOptimizer_list)
            transferInit = self.config['transfer-init']
            pbounds = self.config['pbounds']
            params = pbounds.keys()
            initPoints = []
            if type(transferInit['type']) != list:
                transferInit['type'] = [transferInit['type']]
            for initType in transferInit['type']:
                if initType == 'random':
                    for p in params:
                        initPoints.append(np.random.uniform(low=pbounds[p][0], high=pbounds[p][1], size=transferInit['iters']))
                for i in range(transferInit['iters']):
                    if initType == 'state':
                        stateIndex = np.random.randint(0, len(self.testState_list), 1)[0]
                        self.TransferOptimizer.cost += self.StateOptimizer_list[stateIndex].cost
                        util = UtilityFunction(transferInit['acq'], kappa=transferInit['kappa'], xi=transferInit['xi'])
                        newPoint = self.StateOptimizer_list[stateIndex].suggest(util)
                        gateTarget = self.testGate(**newPoint)
                        stateTarget = self.testState_list[stateIndex](**newPoint)
                        self.StateOptimizer_list[stateIndex].register(newPoint, stateTarget)
                        self.TransferOptimizer.register(newPoint, gateTarget)
                        if i > 2:
                            self.TransferOptimizer.suggest(util)
                        self.TransferOptimizer.cost -= self.StateOptimizer_list[stateIndex].cost
                    elif initType == 'random':
                        newPoint = {}
                        j = 0
                        for p in params:
                            newPoint[p] = initPoints[j][i]
                            j += 1
                        gateTarget = self.testGate(**newPoint)
                        self.TransferOptimizer.register(newPoint, gateTarget)
                    else:
                        raise Exception("Invalid tranfer init type. Choose 'state' or 'random'.")

            for optimization in self.config['transfer'].values():
                if 'refine' in optimization.keys():
                    new_bounds = self.TransferOptimizer.get_new_bounds(optimization['refine'])
                    self.TransferOptimizer.set_bounds(new_bounds)
                if not 'alpha' in optimization.keys():
                    optimization['alpha'] = 0
                    optimization['alpha-decay'] = 1
                    optimization['alpha-delay'] = 0
                if not 'alpha-min' in optimization.keys():
                    optimization['alpha-min'] = 0
                if not 'kappa-min' in optimization.keys():
                    optimization['kappa-min'] = 0
                if 'decay-delay' in optimization.keys() and 'kappa-delay' not in optimization.keys():
                    optimization['kappa-delay'] = optimization['decay-delay']
                if not 'pow' in optimization.keys():
                    optimization['pow'] = 1
                if not 'feedback' in optimization.keys():
                    optimization['feedback'] = 0
                self.TransferOptimizer.maximize(
                    init_points=0,
                    n_iter=optimization['iters'],
                    acq=optimization['acq'],
                    kappa=optimization['kappa'],
                    kappa_decay=optimization['kappa-decay'],
                    kappa_decay_delay=optimization['kappa-delay'],
                    kappa_min=optimization['kappa-min'],
                    alpha=optimization['alpha'],
                    alpha_decay=optimization['alpha-decay'],
                    alpha_decay_delay=optimization['alpha-delay'],
                    alpha_min=optimization['alpha-min'],
                    power=optimization['pow'],
                    feedback=optimization['feedback']
                )

        if self.ControlOptimizer != None:
            for optimization in self.config['control'].values():
                if 'refine' in optimization.keys():
                    new_bounds = self.ControlOptimizer.get_new_bounds(optimization['refine'])
                    self.ControlOptimizer.set_bounds(new_bounds)
                if not 'kappa-min' in optimization.keys():
                    optimization['kappa-min'] = 0
                if 'decay-delay' in optimization.keys() and 'kappa-delay' not in optimization.keys():
                    optimization['kappa-delay'] = optimization['decay-delay']
                self.ControlOptimizer.maximize(
                    init_points=0,
                    n_iter=optimization['iters'],
                    acq=optimization['acq'],
                    kappa=optimization['kappa'],
                    kappa_decay=optimization['kappa-decay'],
                    kappa_decay_delay=optimization['kappa-delay'],
                    kappa_min=optimization['kappa-min'],
                )

    def plot_gps(self, stateIndex='all', stateTitle='State', transferTitle='Gate - Transfer',
                controlTitle='Gate - Standard', show=True, save=False, saveFile='./figures/'):
        if len(self.TransferOptimizer.max['params']) == 1:
            plt.rcParams.update({'font.size': 12})
            plot_bo_1D(self.ControlOptimizer, label='UCB', color='r')
            plot_bo_1D(self.TransferOptimizer, label='Transfer', color='b')
            x = np.linspace(0, 3*np.pi, 1000)
            plt.grid(linestyle='--')
            plt.plot(x, self.testGate(x), label='$-\sin(x)$ - Target Task', color='g')
            plt.plot(x, self.testState_list[0](x), label='$\sin(x)^2$ - Source Task', color='orange')
            plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2, 3*np.pi],
                        ['0', '$\pi$/2', '$\pi$', '$3\pi$/2', '$2\pi$', '$5\pi$/2', '$3\pi$'])
            plt.xlim(0, 3*np.pi)
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            if save:
                plt.savefig(saveFile, bbox_inches='tight', pad_inches=0.1)
        elif len(self.TransferOptimizer.max['params']) == 2:
            if stateIndex == 'all':
                for i in range(len(self.StateOptimizer_list)):
                    plot_bo_2D(self.StateOptimizer_list[i], stateTitle, save=save, saveFile=saveFile)
            elif len(self.StateOptimizer_list) > 0:
                plot_bo_2D(self.StateOptimizer_list[stateIndex], stateTitle, save=save, saveFile=saveFile)
            else:
                raise Exception("Invalid stateIndex: {}, please choose 'all' or an interger".format(stateIndex))
            if self.TransferOptimizer != None:
                plot_bo_2D(self.TransferOptimizer, transferTitle, save=save, saveFile=saveFile)
            if self.ControlOptimizer != None:
                plot_bo_2D(self.ControlOptimizer, controlTitle, save=save, saveFile=saveFile)
        else:
            raise Exception("Cannot plot {}D probelm. Please enter a 1D or 2D problem".format(len(self.TransferOptimizer.max['params'])))
        if show:
            plt.show()

    def get_result(self, format_func=fid_to_infidelity):
        if self.TransferOptimizer != None:
            transferResult = deepcopy(self.TransferOptimizer.data.bestResult)
            transferCosts = deepcopy(self.TransferOptimizer.data.cost)
            for i in range(len(transferResult)):
                transferResult[i] = format_func(transferResult[i])
        else:
            transferResult = None
            transferCosts = None

        if self.ControlOptimizer != None:
            controlResult = deepcopy(self.ControlOptimizer.data.bestResult)
            controlCosts = deepcopy(self.ControlOptimizer.data.cost)
            for i in range(len(controlResult)):
                controlResult[i] = format_func(controlResult[i])
        else:
            controlResult = None
            controlCosts = None

        return transferResult, transferCosts, controlResult, controlCosts


    def plot_result(self, title='', show=True, save=False, saveFile='./figures/infidelity.png'):
        cost = self.config['cost']
        if self.StateOptimizer_list != [] and self.TransferOptimizer != None:
            transferStart = 0
            for optimization in self.config['state'].values():
                transferStart += (optimization['iters'] + self.config['state-init']) * self.config['cost']['state'] \
                    * len(self.config['input-states'])
            transferEnd = transferStart + (self.config['transfer-init']['iters'] * len(self.config['transfer-init']['type']) \
                * (self.config['cost']['gate'] + (self.config['cost']['state'])))

        transferResult, transferCosts, controlResult, controlCosts = self.get_result()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if transferCosts != None:
            ax.plot(transferCosts, transferResult, label='With Transfer', color='b')
            ax.axvline(x=transferStart, color='g', linestyle='--', label='Transfer Start')
            ax.axvline(x=transferEnd, color='orange', linestyle='--', label='Transfer End')
        if controlCosts != None:
            ax.plot(controlCosts, controlResult, label='No Transfer', color='r')
        ax.set_xlabel('Cost')
        ax.set_ylabel('Best Infidelity')
        ax.grid('--')
        ax.set_yscale('log')
        fig.legend()
        if save:
            plt.savefig(saveFile)
        if show:
            plt.show()


