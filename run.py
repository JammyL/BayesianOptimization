from problems import hadamard
from qutip.qip.operations import snot
import qutip as qt
import numpy as np
import pickle

initialState = qt.basis(2, 0)

transferResults = []
controlResults = []

for _ in range(1):
    p = hadamard(initialState)
    p.default_opt()
    p.plot_result()
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

pickle.dump( resultsToPickle, open( "test_2.pickle", "wb" ) )
