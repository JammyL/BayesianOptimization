from ..problem import problem
from .two_q_generators import *
from qutip.qip.operations import snot
import qutip as qt

from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor

spins = []

for i in range(1, 3):
    spins.append(qt.basis(2,0))

initial_state = tensor(spins)

QC = QubitCircuit(2)
QC.add_gate("RX", targets=0, arg_value= 0.5)
QC.add_gate("RY", targets=1, arg_value= 0.1)
QC.add_gate("CNOT", targets = 1, controls = 0)
QC.add_gate("RY", targets = 0, arg_value = 0.26127)
QC.add_gate("RX", targets = 1, arg_value= 1.3942948)
U_list = QC.propagators()

TargetGate = gate_sequence_product(U_list)
TargetState = TargetGate * initial_state

class twoQubitCircuit(problem):
    def __init__(self, initialState_list=[initial_state], targetGate=TargetGate, configPath='./problems/two_qubits/two_q_config.yaml', verbose=1):
        targetState_list = [targetGate * initialState for initialState in initialState_list]
        testState_list = [N2qubitStateFunc(initialState_list[i], targetState_list[i]) for i in range(len(initialState_list))]
        testGate = N2qubitGateFunc(targetGate)
        problem.__init__(self, testState_list=testState_list, testGate=testGate, configPath=configPath, verbose=verbose)