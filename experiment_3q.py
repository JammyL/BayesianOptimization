from problems import ThreeQubitCircuit
from qutip.qip.operations import snot
import qutip as qt
import numpy as np
import pickle

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor

spins = []

for i in range(1, 4):
    globals()['spin%s' % i] = qt.basis(2,0)
    spins.append(qt.basis(2,0))
    
initial_state = tensor(spins)

#initialState = qt.basis(2, 0)

transferResults = []
controlResults = []

for _ in range(20):
    p = ThreeQubitCircuit(initial_state)
    p.default_opt()
    tResult, tCost, cResult, cCost = p.get_result()
    transferResults.append(tResult)
    controlResults.append(cResult)
    print("Transfer: ", tResult[-1])
    print("Control: ", cResult[-1])

resultsToPickle = {
    'transfer': np.array(transferResults),
    'transfer_cost': tCost,
    'control': np.array(controlResults),
    'control_cost': cCost,
}
pickle.dump( resultsToPickle, open( "test.pickle", "wb" ) )
