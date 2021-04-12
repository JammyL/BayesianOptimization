from ..problem import problem
from .three_q_generators import *
from numpy import pi
from qutip.qip.operations import snot
from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor
import qutip as qt

spins = []

for i in range(1, 4):
    spins.append(qt.basis(2,0))

initial_state = tensor(spins)

QC = QubitCircuit(3)
QC.add_gate("RX", targets=0, arg_value= 0.5)
QC.add_gate("RX", targets=1, arg_value= 0.1)
QC.add_gate("RX", targets=2, arg_value= 0.2223472)
QC.add_gate("CNOT", targets = 1, controls = 0)
QC.add_gate("CNOT", targets = 2, controls = 0)
QC.add_gate("RX", targets = 0, arg_value = 0.26127)
QC.add_gate("RX", targets = 1, arg_value= 1.3942948)
QC.add_gate("RX", targets = 1, arg_value= 0.4378)
U_list = QC.propagators()
TargetGate = gate_sequence_product(U_list)

#defining a circuit with noise

QC1 = QubitCircuit(3)
QC1.user_gates = {"noisy1": noisy_unitary1,
                "noisy2": noisy_unitary2,
                "noisy3": noisy_unitary3}

QC1.add_gate("RX", targets=0, arg_value= 0.5)
QC1.add_gate("noisy3", targets=0, arg_value = 0.1)
QC1.add_gate("noisy3", targets=1, arg_value = 0.1)
QC1.add_gate("noisy3", targets=2, arg_value = 0.1)
QC1.add_gate("RX", targets=1, arg_value= 0.1)
QC1.add_gate("RX", targets=2, arg_value= 0.2223472)
QC1.add_gate("CNOT", targets = 1, controls = 0)
QC1.add_gate("CNOT", targets = 2, controls = 0)
QC1.add_gate("noisy3", targets=0, arg_value = 0.1)
QC1.add_gate("noisy3", targets=1, arg_value = 0.1)
QC1.add_gate("noisy3", targets=2, arg_value = 0.1)
QC1.add_gate("RX", targets = 0, arg_value = 0.26127)
QC1.add_gate("RX", targets = 1, arg_value= 1.3942948)
QC1.add_gate("RX", targets = 1, arg_value= 0.4378)
QC1.add_gate("noisy3", targets=0, arg_value = 0.1)
QC1.add_gate("noisy3", targets=1, arg_value = 0.1)
QC1.add_gate("noisy3", targets=2, arg_value = 0.1)
U_list1 = QC1.propagators()
TestGate = gate_sequence_product(U_list1)
print(TestGate)

class threeQubitCircuit(problem):
    def __init__(self, initialState_list=[initial_state], targetGate=TargetGate, configPath='./problems/three_qubits/three_q_config.yaml', verbose=2):
        targetState_list = [targetGate * initialState for initialState in initialState_list]
        testState_list = [N3qubitStateFunc(initialState_list[i], targetState_list[i]) for i in range(len(initialState_list))]
        testGate = N3qubitGateFunc(targetGate)
        problem.__init__(self, testState_list=testState_list, testGate=testGate, configPath=configPath, verbose=verbose)