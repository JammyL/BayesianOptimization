from helpers import *
from bayes_opt import BayesianOptimization, TargetBayesianOptimization
from qutip.qip.operations import snot

import numpy as np
import qutip as qt

initialState = qt.basis(2, 0)

targetOp = snot(1)
targetState = targetOp * initialState

testOpParams = singleOpFunc(targetOp)
testStateParams = singleStateFunc(initialState, targetState)

pbounds = {'a1': (0, 2), 'a2': (-2, 2)}

a1_init_points = np.random.uniform(low=pbounds['a1'][0], high=pbounds['a1'][1], size=5)
a2_init_points = np.random.uniform(low=pbounds['a2'][0], high=pbounds['a2'][1], size=5)

State_optimizer = BayesianOptimization(
    f=testStateParams,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
)
Op_optimizer = BayesianOptimization(
    f=testOpParams,
    pbounds=pbounds,
)

for i in range(len(a1_init_points)):
    nextPoint = {'a1': a1_init_points[i], 'a2': a2_init_points[i]}
    stateTarget = testStateParams(nextPoint['a1'], nextPoint['a2'])
    OpTarget = testOpParams(nextPoint['a1'], nextPoint['a2'])
    Op_optimizer.register(nextPoint, OpTarget)
    State_optimizer.register(nextPoint, stateTarget)

############### Initial State Opt ###############
State_optimizer.maximize(
    init_points=0,
    n_iter=10,
    kappa=25,
)
State_optimizer.maximize(
    init_points=0,
    n_iter=25,
    kappa=5,
)
State_optimizer.maximize(
    init_points=0,
    n_iter=25,
    kappa=1,
)
print(State_optimizer.max)
print(1 - State_optimizer.max['target'])
#############################################

TransferOp_optimizer = TargetBayesianOptimization(
    f=testOpParams,
    pbounds=pbounds,
    source_gp=State_optimizer._gp
)

for i in range(len(a1_init_points)):
    nextPoint = {'a1': a1_init_points[i], 'a2': a2_init_points[i]}
    TransferOpTarget = testOpParams(nextPoint['a1'], nextPoint['a2'])
    TransferOp_optimizer.register(nextPoint, TransferOpTarget)

TransferOp_optimizer.maximize(
    init_points=0,
    n_iter=5,
)

print(TransferOp_optimizer.max)
print(1 - TransferOp_optimizer.max['target'])

Op_optimizer.maximize(
    init_points=0,
    n_iter=10,
)

print(Op_optimizer.max)
print(1 - Op_optimizer.max['target'])
