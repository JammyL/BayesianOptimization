from ..problem import problem
from .three_q_yz_generators import *
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
QC.add_gate("RY", targets=0, arg_value=pi/2)
QC.add_gate("RZ", targets=1, arg_value=pi/4)
QC.add_gate("RY", targets=2, arg_value=pi/2)
QC.add_gate("CNOT", targets=1, controls=0)
QC.add_gate("CNOT", targets=2, controls=0)
QC.add_gate("RZ", targets=0, arg_value=pi/3)
QC.add_gate("RY", targets=1, arg_value=pi/2)
QC.add_gate("RZ", targets=2, arg_value=pi/4)
U_list = QC.propagators()

TargetGate = gate_sequence_product(U_list)

class threeQubitCircuitYZ(problem):
    def __init__(self, initialState_list=[initial_state], targetGate=TargetGate, epsilon=0, configPath='./problems/three_qubits/three_q_config.yaml', verbose=2):
        targetState_list = [targetGate * initialState for initialState in initialState_list]
        testState_list = [N3qubitStateFunc(initialState_list[i], targetState_list[i], epsilon) for i in range(len(initialState_list))]
        testGate = N3qubitGateFunc(targetGate, epsilon)
        problem.__init__(self, testState_list=testState_list, testGate=testGate, configPath=configPath, verbose=verbose)