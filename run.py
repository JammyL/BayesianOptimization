from problems import threeQubitCircuit
import qutip as qt
import numpy as np
import pickle
import sys

args = sys.argv[1:]
if len(args) < 2:
    configPath = './configs/three_q/delay/config_1.yaml'
    outputPath = 'results_10.pickle'
else:
    configPath = args[0]
    outputPath = args[1]

controlResults = []
transferResults = []

for _ in range(1000):
    p = threeQubitCircuit(configPath=configPath)
    p.default_opt()
    tResult, tCost, cResult, cCost = p.get_result()
    if p.ControlOptimizer != None:
        controlResults.append(cResult)
    if p.TransferOptimizer != None:
        transferResults.append(tResults)

resultsToPickle = {
    'config': p.config,
    'control': np.array(controlResults),
    'control_cost': cCost,
    'transfer': np.array(transferResults),
    'transfer_cost': tCost,
}

pickle.dump( resultsToPickle, open( outputPath, "wb" ) )