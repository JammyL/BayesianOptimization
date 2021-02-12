from problems import hadamard, threeQubitCircuit
from qutip.tensor import tensor
import qutip as qt
import numpy as np
import pickle
import sys
import yaml

args = sys.argv[1:]
if len(args) < 3:
    configPath = './configs/three_q/bad_path/config_1.yaml'
    outputPath = 'results_10.pickle'
    numberStates = 1
else:
    configPath = args[0]
    outputPath = args[1]
    numberStates = int(args[2])

initial_state_list = []
for i in range(numberStates):
    spins = []
    bits = [int(x) for x in '{:03b}'.format(i)]
    for b in bits:
        spins.append(qt.basis(2,b))
    initial_state_list.append(tensor(spins))

controlResults = []
transferResults = []

for _ in range(1000):
    p = threeQubitCircuit(initialState_list=initial_state_list, configPath=configPath,)
    p.default_opt()
    tResult, tCost, cResult, cCost = p.get_result()
    if p.ControlOptimizer != None:
        controlResults.append(cResult)
    if p.TransferOptimizer != None:
        transferResults.append(tResult)

resultsToPickle = {
    'config': p.config,
    'control': np.array(controlResults),
    'control_cost': cCost,
    'transfer': np.array(transferResults),
    'transfer_cost': tCost,
}

pickle.dump( resultsToPickle, open( outputPath, "wb" ) )
