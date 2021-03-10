from ..problem import problem
from .two_q_generators import *
from numpy import pi

from qutip.qip.operations import snot
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor

import qutip as qt

spins = []
for _ in range(2):
    spins.append(qt.basis(2,0))
initial_state = tensor(spins)

QC = QubitCircuit(2)
QC.add_gate("RX", targets=0, arg_value= pi/2)
QC.add_gate("RY", targets=1, arg_value= pi/12)
QC.add_gate("CNOT", targets = 1, controls = 0)
QC.add_gate("RY", targets = 0, arg_value = pi/5)
QC.add_gate("RX", targets = 1, arg_value= pi/7)
U_list = QC.propagators()

TargetGate = gate_sequence_product(U_list)

class twoQubitCircuit(problem):
    def __init__(self, initialState_list=[initial_state], targetGate=TargetGate, configPath='./problems/two_qubits/two_q_config.yaml', verbose=2):
        targetState_list = [targetGate * initialState for initialState in initialState_list]
        testState_list = [N2qubitStateFunc(initialState_list[i], targetState_list[i]) for i in range(len(initialState_list))]
        testGate = N2qubitGateFunc(targetGate)
        problem.__init__(self, testState_list=testState_list, testGate=testGate, configPath=configPath, verbose=verbose)