from helpers import *
from bayes_opt import BayesianOptimization, TargetBayesianOptimization, UtilityFunction, MultiUtilityFunction
from bayes_opt import SequentialDomainReductionTransformer
from qutip.qip.operations import snot

import pickle
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

initialState = qt.basis(2, 0)

targetOp = snot(1)
targetState = targetOp * initialState

testOpParams = singleOpFunc(targetOp)
testStateParams = singleStateFunc(initialState, targetState)

initPoints = 5
StateCost = 1
OpCost = 10
stateIters = 40
transferIters = 10
transferRegularIters = 10
controlIters = 25

pbounds = {'a1': (0, 2), 'a2': (-2, 2)}

a1_init_points = np.random.uniform(low=pbounds['a1'][0], high=pbounds['a1'][1], size=initPoints)
a2_init_points = np.random.uniform(low=pbounds['a2'][0], high=pbounds['a2'][1], size=initPoints)

State_optimizer = BayesianOptimization(
    f=testStateParams,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    cost=StateCost,
    random_state=2,
)

Control_optimizer = BayesianOptimization(
    f=testOpParams,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    cost=OpCost,
    random_state=1,
    # bounds_transformer= SequentialDomainReductionTransformer(),
)

for i in range(len(a1_init_points)):
    nextPoint = {'a1': a1_init_points[i], 'a2': a2_init_points[i]}
    stateTarget = testStateParams(nextPoint['a1'], nextPoint['a2'])
    opTarget = testOpParams(nextPoint['a1'], nextPoint['a2'])
    State_optimizer.register(nextPoint, stateTarget)
    Control_optimizer.register(nextPoint, opTarget)

print(State_optimizer.max)
print(1 - State_optimizer.max['target'])

State_optimizer.maximize(
    init_points=0,
    n_iter=stateIters,
    kappa=20,
    # kappa_decay=0.5
)

Transfer_optimizer = TargetBayesianOptimization(
    f=testOpParams,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    source_gp=State_optimizer._gp,
    cost=OpCost,
    random_state=1,
    # bounds_transformer= SequentialDomainReductionTransformer(),
)

Transfer_optimizer.transferData(State_optimizer)

for i in range(len(a1_init_points)):
    nextPoint = {'a1': a1_init_points[i], 'a2': a2_init_points[i]}
    target = testOpParams(nextPoint['a1'], nextPoint['a2'])
    Transfer_optimizer.register(nextPoint, target)

Transfer_optimizer.maximize(
    init_points=0,
    n_iter=transferIters,
    # kappa=4
)

Transfer_optimizer.maximize(
    init_points=0,
    n_iter=transferRegularIters,
    acq='ucb',
    kappa=1,
    kappa_decay=0.5,
)

Control_optimizer.maximize(
    init_points=0,
    n_iter=controlIters,
    kappa=1,
    kappa_decay=0.5,
    kappa_decay_delay=10,
)

bestTransferInf = Transfer_optimizer.data.bestResult
bestControlInf = Control_optimizer.data.bestResult
for i in range(len(bestTransferInf)):
    bestTransferInf[i] = 1 - bestTransferInf[i]

for i in range(len(bestControlInf)):
    bestControlInf[i] = 1 - bestControlInf[i]

# pickle_obj = {
#     "transfer_inf": bestTransferInf,
#     "transfer_best": Transfer_optimizer.data.bestPoints,
#     "transfer_cost": Transfer_optimizer.data.cost,
#     "transfer_start": stateIters * StateCost,
#     "control_inf": bestControlInf,
#     "control_best": Control_optimizer.data.bestPoints,
#     "control_cost": Control_optimizer.data.cost,
# }

print("State:", State_optimizer.max)
print("Transfer:", bestTransferInf[-1])
print("Control:", bestControlInf[-1])

# pickle.dump( pickle_obj, open( "test_2.pickle", "wb" ) )

plot_bo(State_optimizer, "State")
plot_bo(Transfer_optimizer, "Transfer")
plot_bo(Control_optimizer, "Control")

transferPoint = (initPoints + stateIters) * StateCost
regularBayesPoint = transferPoint + ((initPoints + transferIters) * OpCost)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Control_optimizer.data.cost, bestControlInf, label='No Transfer', color='r')
ax.plot(Transfer_optimizer.data.cost, bestTransferInf, label='With Transfer', color='b')
ax.set_xlabel('Cost')
ax.set_ylabel('Best Infidelity')
ax.grid('--')
ax.set_yscale('log')
ax.axvline(x=transferPoint, color='g', linestyle='--', label='Transfer Point')
plt.axvline(x=regularBayesPoint, color='orange', linestyle='--', label='Switch to UCB')
fig.legend()

plt.show()
