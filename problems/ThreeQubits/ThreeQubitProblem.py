from ..problem import problem
from .ThreeQgenerators import *
from qutip.qip.operations import snot
import qutip as qt

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor

spins = []

for i in range(1, 4):
    globals()['spin%s' % i] = qt.basis(2,0)
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
TargetState = TargetGate * initial_state

class ThreeQubitCircuit(problem):
    def __init__(self, initialState = initial_state, targetGate = TargetGate, configPath='./problems/ThreeQubits/config3QG.yaml', verbose=2):
        targetState = targetGate * initialState
        testState_list = [N3qubitStateFunc(initialState, targetState)]
        testGate = N3qubitGateFunc(targetGate)
        problem.__init__(self, testState_list=testState_list, testGate=testGate, configPath=configPath, verbose=verbose)