from problems import threeQubitCircuit
from qutip.qip.operations import snot
import qutip as qt
import numpy as np
import pickle

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor

spins_0 = []
spins_1 = []

for i in range(1, 4):
    spins_0.append(qt.basis(2,0))
    spins_1.append(qt.basis(2,1))

state_0 = tensor(spins_0)
state_1 = tensor(spins_1)
initial_state_list = [state_0]


#initialState = qt.basis(2, 0)

transferResults = []
controlResults = []

for _ in range(1):
    p = threeQubitCircuit(initialState_list=initial_state_list)
    p.default_opt()
    p.plot_result()
    tResult, tCost, cResult, cCost = p.get_result()
    transferResults.append(tResult)
    controlResults.append(cResult)
    print("Transfer:", p.TransferOptimizer.max['params'])
    print("Transfer: ", tResult[-1])
    print("Control:", p.ControlOptimizer.max['params'])
    print("Control: ", cResult[-1])

resultsToPickle = {
    'transfer': np.array(transferResults),
    'transfer_cost': tCost,
    'control': np.array(controlResults),
    'control_cost': cCost,
}
pickle.dump( resultsToPickle, open( "test.pickle", "wb" ) )
