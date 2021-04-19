from problems import hadamard, hadamardRandomState, twoQubitCircuit, threeQubitCircuit, threeQubitCircuitYZ, oneQubitCircuit, threeQubitCircuitRandom
from qutip.qip.operations import snot
from qutip.tensor import tensor
from qutip.random_objects import rand_ket
import qutip as qt
import numpy as np
import logging
import pickle
import sys
import yaml

args = sys.argv[1:]
if len(args) < 2:
    #configPath = './configs/three_q/general_tests/3q_c1.yaml'
    #configPath = './problems/one_qubit_1/one_q_config.yaml'
    configPath = './configs/general/hadamard_great.yaml'
    configPath = './configs/general/three_q_yz.yaml'
    configPath = './problems/three_qubits_random/three_q_config.yaml'
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
    'hadamard_random_state': hadamardRandomState,
    'two_q': twoQubitCircuit,
    'three_q': threeQubitCircuit,
    'three_q_yz': threeQubitCircuitYZ,
    'one_q_1': oneQubitCircuit,
    'three_q_rand': threeQubitCircuitRandom,
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
        if qubit == 'r':
            qubit_list.append(rand_ket(2))
        elif qubit == '+':
            qubit_list.append(snot(1) * qt.basis(2,0))
        elif qubit == '-':
            qubit_list.append(snot(1) * qt.basis(2,1))
        elif qubit == '0':
            qubit_list.append(qt.basis(2,0))
        elif qubit == '1':
            qubit_list.append(qt.basis(2,1))
        elif qubit == 'R':
            qubit_list.append(qt.basis(2,0)*1/np.sqrt(2) + qt.basis(2,1)*1j/np.sqrt(2))
        elif qubit == 'L':
            qubit_list.append(qt.basis(2,0)*1/np.sqrt(2) - qt.basis(2,1)*1j/np.sqrt(2))
        else:
            raise Exception("Invalid qubit choice: %s, please choose '0', '1', '+', or '-'".format(qubit))
    initial_state_list.append(tensor(qubit_list))

if len(initial_state_list) == 0 and 'state' in config.keys():
    raise Exception("No states specified, please specify 'input-states:' in {}".format(configPath))

controlResults = []
transferResults = []

for i in range(63):
    try:
        p = problem(initialState_list=initial_state_list, configPath=configPath, verbose=1)
        p.default_opt()
        #p.plot_result()
        #p.plot_gps(save=True)
        tResult, tCost, cResult, cCost = p.get_result()
        if p.ControlOptimizer != None:
            controlResults.append(cResult)
        if p.TransferOptimizer != None:
            transferResults.append(tResult)
    except Exception as e:
        logging.exception('Error in iteration {}'.format(i))

resultsToPickle = {
    'config': p.config,
    'control': np.array(controlResults),
    'control_cost': cCost,
    'transfer': np.array(transferResults),
    'transfer_cost': tCost,
}

pickle.dump( resultsToPickle, open( outputPath, "wb" ) )
