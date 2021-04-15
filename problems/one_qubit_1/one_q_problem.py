from ..problem import problem
#from one_q_generators.py import *
from qutip.qip.operations import snot
import qutip as qt
from numpy import pi
from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor
import qutip as qt
import numpy as np

def calcStateFidelity(finalState, targetState):
    finalState = finalState.dag()
    return abs(targetState.overlap(finalState))**2

def calcGateFidelity(finalGate, targetGate):
    product = finalGate * targetGate.dag()
    return (abs(product[0][0][0] + product[1][0][1])**2) / 4

"""
def singleStateFunc(initialState, targetState):
    def testStateParams(a1, a2):
        R_z = (qt.sigmaz() * 1j * a1).expm()
        R_y = (qt.sigmay() * 1j * a2).expm()
        propagator = (-1j) * R_z * R_y
        propagatedState = propagator * initialState
        return calcStateFidelity(propagatedState, targetState)
    return testStateParams

def singleGateFunc(targetGate):
    def testGateParams(a1, a2):
        R_z = (qt.sigmaz() * 1j * a1).expm()
        R_y = (qt.sigmay() * 1j * a2).expm()
        propagator = (-1j) * R_z * R_y
        return calcGateFidelity(propagator, targetGate)
    return testGateParams
"""

def calcGateFidelityN(finalGate, targetGate, N):
    product = finalGate * targetGate.dag()
    trace = np.trace(product)
    return (abs(trace)**2)/(4**N)

def singleGateFunc(targetGate):
    def testGateParams(a1, a2):
        QC = QubitCircuit(1)
        QC.add_gate("RX", targets=0, arg_value= a1)
        QC.add_gate("RY", targets=0, arg_value= a2)
        U_list = QC.propagators()
        finalGate = gate_sequence_product(U_list)
        return calcGateFidelityN(finalGate, targetGate, 1)
    return testGateParams

def singleStateFunc(initialState, targetState):
    def testStateParams(a1, a2):
        QC = QubitCircuit(1)
        QC.add_gate("RX", targets=0, arg_value= a1)
        QC.add_gate("RY", targets=0, arg_value= a2)
        U_list = QC.propagators()
        finalGate = gate_sequence_product(U_list)
        finalState = finalGate * initialState
        return calcStateFidelity(targetState, finalState)
    return testStateParams

def generatePropagator(a1, a2):
    R_z = (qt.sigmaz() * 1j * a1).expm()
    R_y = (qt.sigmay() * 1j * a2).expm()
    return (-1j) * R_z * R_y


spins = []

for i in range(1, 2):
    spins.append(qt.basis(2,0))

initial_state = tensor(spins)


tp = np.random.uniform(low=0, high=3.14, size=(2))
print(tp) #delete later
QC = qt.QubitCircuit(1)
QC.add_gate("RX", targets=0, arg_value= tp[0])
QC.add_gate("RY", targets = 0, arg_value= tp[1])
U_list = QC.propagators()

TargetGate = gate_sequence_product(U_list)

class oneQubitCircuit(problem):
    def __init__(self, initialState_list, targetGate=TargetGate, configPath='./problems/hadamard/hadamard_config.yaml', verbose=2):
        targetState_list = [targetGate * initialState for initialState in initialState_list]
        testState_list = [singleStateFunc(initialState_list[i], targetState_list[i]) for i in range(len(initialState_list))]
        testGate = singleGateFunc(targetGate)
        problem.__init__(self, testState_list=testState_list, testGate=testGate, configPath=configPath, verbose=verbose)
