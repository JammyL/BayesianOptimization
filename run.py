from problems import hadamard, threeQubitCircuit
from qutip.qip.operations import snot
from qutip.tensor import tensor
import qutip as qt
import numpy as np
import pickle
import sys
import yaml

args = sys.argv[1:]
if len(args) < 2:
    configPath = './problems/three_qubits/three_q_config.yaml'
    outputPath = 'results_10.pickle'
else:
    configPath = args[0]
    outputPath = args[1]

with open(configPath) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)

problem_dict = {
    'hadamard': hadamard,
    'three_q': threeQubitCircuit,
}

if 'problem' in config.keys():
    problem = problem_dict[config['problem']]
else:
    raise Exception("No problem specified in config. Please specify a problem e.g. 'problem: hadamard'")

initial_state_list = []
if 'input-states' in config.keys():
    input_states = config['input-states']
else:
    input_states = []

for state in input_states:
    qubit_list = []
    for qubit in reversed(state):
        if qubit == '+':
            qubit_list.append(snot(1) * qt.basis(2,0))
        elif qubit == '-':
            qubit_list.append(snot(1) * qt.basis(2,1))
        elif qubit == '0':
            qubit_list.append(qt.basis(2,0))
        elif qubit == '1':
            qubit_list.append(qt.basis(2,1))
        else:
            raise Exception("Invalid qubit choice: %s, please choose '0', '1', '+', or '-'".format(qubit))
    initial_state_list.append(tensor(qubit_list))

controlResults = []
transferResults = []

for _ in range(1):
    p = problem(initialState_list=initial_state_list, configPath=configPath)
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
