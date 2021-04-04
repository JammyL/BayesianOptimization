import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor

def noisyUnitaryPoorStats(err): #definition by poor stats paper
    n = np.random.rand(3)
    n = n / np.linalg.norm(n)
    if err > 1:
        err = 1
    if err < 0:
        err = 0
    H = (np.sqrt(1-np.power(err,2))*qt.qeye(2)) + (1j * err* (n[0] * qt.sigmax() + n[1] * qt.sigmay() + n[2] * qt.sigmaz()))
    return H

def calcStateFidelity(finalState, targetState):
    finalState = finalState.dag()
    return abs(targetState.overlap(finalState))**2

def calcGateFidelityN(finalGate, targetGate, N):
    product = finalGate * targetGate.dag()
    trace = np.trace(product)
    return (abs(trace)**2)/(4**N)

def N3qubitGateFunc(targetGate, epsilon):
    def testGateParamsPure(a1, a2, a3, a4, a5, a6):
        QCP = QubitCircuit(3)
        QCP.add_gate("RY", targets=0, arg_value=a1)
        QCP.add_gate("RZ", targets=1, arg_value=a2)
        QCP.add_gate("RY", targets=2, arg_value=a3)
        QCP.add_gate("CNOT", targets=1, controls=0)
        QCP.add_gate("CNOT", targets=2, controls=0)
        QCP.add_gate("RZ", targets=0, arg_value=a4)
        QCP.add_gate("RY", targets=1, arg_value=a5)
        QCP.add_gate("RZ", targets=2, arg_value=a6)
        U_list_P = QCP.propagators()
        finalGate_P = gate_sequence_product(U_list_P)
        return calcGateFidelityN(finalGate_P, targetGate, 3)
    if epsilon != 0:
        def testGateParamsNoise(a1, a2, a3, a4, a5, a6):
            QCN = QubitCircuit(3)
            noise = np.random.normal(0, epsilon, 9)
            QCN.user_gates = {"noise": noisyUnitaryPoorStats}
            QCN.add_gate("noise", targets=0, arg_value = noise[0])
            QCN.add_gate("noise", targets=1, arg_value = noise[1])
            QCN.add_gate("noise", targets=2, arg_value = noise[2])
            QCN.add_gate("RY", targets=0, arg_value=a1)
            QCN.add_gate("RZ", targets=1, arg_value=a2)
            QCN.add_gate("RY", targets=2, arg_value=a3)
            QCN.add_gate("CNOT", targets=1, controls=0)
            QCN.add_gate("CNOT", targets=2, controls=0)
            QCN.add_gate("noise", targets=0, arg_value = noise[0])
            QCN.add_gate("noise", targets=1, arg_value = noise[1])
            QCN.add_gate("noise", targets=2, arg_value = noise[2])
            QCN.add_gate("RZ", targets=0, arg_value=a4)
            QCN.add_gate("RY", targets=1, arg_value=a5)
            QCN.add_gate("RZ", targets=2, arg_value=a6)
            QCN.add_gate("noise", targets=0, arg_value = noise[0])
            QCN.add_gate("noise", targets=1, arg_value = noise[1])
            QCN.add_gate("noise", targets=2, arg_value = noise[2])
            U_list_N = QCN.propagators()
            finalGate_N = gate_sequence_product(U_list_N)
            return calcGateFidelityN(finalGate_N, targetGate, 3)
        return (testGateParamsPure, testGateParamsNoise)
    return (testGateParamsPure, testGateParamsPure)

def N3qubitStateFunc(initialState, targetState, epsilon):
    def testStateParamsPure(a1, a2, a3, a4, a5, a6):
        QCP = QubitCircuit(3)
        QCP.add_gate("RY", targets=0, arg_value=a1)
        QCP.add_gate("RZ", targets=1, arg_value=a2)
        QCP.add_gate("RY", targets=2, arg_value=a3)
        QCP.add_gate("CNOT", targets=1, controls=0)
        QCP.add_gate("CNOT", targets=2, controls=0)
        QCP.add_gate("RZ", targets=0, arg_value=a4)
        QCP.add_gate("RY", targets=1, arg_value=a5)
        QCP.add_gate("RZ", targets=2, arg_value=a6)
        U_list_P = QCP.propagators()
        finalGate_P = gate_sequence_product(U_list_P)
        finalState_P = finalGate_P * initialState
        return calcStateFidelity(targetState, finalState_P)
    if epsilon != 0:
        def testStateParamsNoise(a1, a2, a3, a4, a5, a6):
            QCN = QubitCircuit(3)
            noise = np.random.normal(0, epsilon, 9)
            QCN.user_gates = {"noise": noisyUnitaryPoorStats}
            QCN.add_gate("noise", targets=0, arg_value = noise[0])
            QCN.add_gate("noise", targets=1, arg_value = noise[1])
            QCN.add_gate("noise", targets=2, arg_value = noise[2])
            QCN.add_gate("RY", targets=0, arg_value=a1)
            QCN.add_gate("RZ", targets=1, arg_value=a2)
            QCN.add_gate("RY", targets=2, arg_value=a3)
            QCN.add_gate("CNOT", targets=1, controls=0)
            QCN.add_gate("CNOT", targets=2, controls=0)
            QCN.add_gate("noise", targets=0, arg_value = noise[0])
            QCN.add_gate("noise", targets=1, arg_value = noise[1])
            QCN.add_gate("noise", targets=2, arg_value = noise[2])
            QCN.add_gate("RZ", targets=0, arg_value=a4)
            QCN.add_gate("RY", targets=1, arg_value=a5)
            QCN.add_gate("RZ", targets=2, arg_value=a6)
            QCN.add_gate("noise", targets=0, arg_value = noise[0])
            QCN.add_gate("noise", targets=1, arg_value = noise[1])
            QCN.add_gate("noise", targets=2, arg_value = noise[2])
            U_list = QCN.propagators()
            finalGate_N = gate_sequence_product(U_list)
            finalState_N = finalGate_N * initialState
            return calcStateFidelity(targetState, finalState_N)
        return (testStateParamsPure, testStateParamsNoise)
    return (testStateParamsPure, testStateParamsPure)