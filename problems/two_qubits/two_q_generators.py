import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import gate_sequence_product

def calcStateFidelity(finalState, targetState):
    finalState = finalState.dag()
    return abs(targetState.overlap(finalState))**2

def calcGateFidelityN(finalGate, targetGate, N):
    product = finalGate * targetGate.dag()
    trace = np.trace(product)
    return (abs(trace)**2)/(4**N)

def N2qubitGateFunc(targetGate):
    def testGateParams(a1, a2, a3, a4):
        QC = QubitCircuit(2)
        QC.add_gate("RX", targets=0, arg_value= a1)
        QC.add_gate("RY", targets=1, arg_value= a2)
        QC.add_gate("CNOT", targets = 1, controls = 0)
        QC.add_gate("RY", targets = 0, arg_value = a3)
        QC.add_gate("RX", targets = 1, arg_value= a4)
        U_list = QC.propagators()
        finalGate = gate_sequence_product(U_list)
        return calcGateFidelityN(finalGate, targetGate, 2)
    return testGateParams

def N2qubitStateFunc(initialState, targetState):
    def testStateParams(a1, a2, a3, a4):
        QC = QubitCircuit(2)
        QC.add_gate("RX", targets=0, arg_value= a1)
        QC.add_gate("RY", targets=1, arg_value= a2)
        QC.add_gate("CNOT", targets = 1, controls = 0)
        QC.add_gate("RY", targets = 0, arg_value = a3)
        QC.add_gate("RX", targets = 1, arg_value= a4)
        U_list = QC.propagators()
        finalGate = gate_sequence_product(U_list)
        finalState = finalGate * initialState
        return calcStateFidelity(targetState, finalState)
    return testStateParams