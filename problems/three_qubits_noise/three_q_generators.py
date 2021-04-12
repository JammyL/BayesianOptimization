import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cmath

from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.operations import gate_sequence_product
from qutip.tensor import tensor

def calcGateFidelityN(finalGate, targetGate, N):
    product = finalGate * targetGate.dag()
    trace = np.trace(product)
    return (abs(trace)**2)/(4**N)

#defining noisy unitaries
def noisy_unitary1(err): #taking exponent of hermitian matrix
    r1 = np.random.random()
    r2 = np.random.random()
    r3 = np.random.random()
    total = r1**2 + r2**2 + r3**2
    r1 = np.sqrt(r1**2/total)
    r2 = np.sqrt(r2**2/total)
    r3 = np.sqrt(r3**2/total)
    H = np.sqrt(1-err**2)*qt.qeye(2) + err* (r1 * qt.sigmax() + r2 * qt.sigmay() + r3 * qt.sigmaz())
    A = (-1j) * H
    U = A.expm()
    return U

def noisy_unitary2(err): #definition by poor stats paper
    r1 = np.random.random()
    r2 = np.random.random()
    r3 = np.random.random()
    total = r1**2 + r2**2 + r3**2
    r1 = np.sqrt(r1**2/total)
    r2 = np.sqrt(r2**2/total)
    r3 = np.sqrt(r3**2/total)
    H = np.sqrt(1-err**2)*qt.qeye(2) + 1j * err* (r1 * qt.sigmax() + r2 * qt.sigmay() + r3 * qt.sigmaz())
    return H

#testing unitarity
U2 = noisy_unitary2(0.1)
U2d = U2.dag()
print(U2, U2*U2d, U2d*U2, U2.type)

def noisy_unitary3(err): #definition by wiki, most general form for unitary transformation
    b = err
    a = np.sqrt(1-err**2)
    rs = np.random.normal(0, 0.01, 3)
    rs = np.absolute(rs)

    r1 = rs[0]
    r2 = rs[1]
    r3 = rs[2]

    ex = np.cos(r3) + 1j * np.sin(r3)
    a1 = a * (np.cos(r1) + 1j * np.sin(r1))
    a2 = a * (np.cos(r1) - 1j * np.sin(r1))
    b1 = b * (np.cos(r2) + 1j * np.sin(r2))
    b2 = b * (np.cos(r2) - 1j * np.sin(r2))
    mat = np.array([[a1,   b1],
                    [-ex*b2, ex*a2]])
    return qt.Qobj(mat, dims=[[2], [2]])

QC = QubitCircuit(3) #noiseless circuit
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

#defining a circuit with noise based on noisy1
QC1 = QubitCircuit(3)
QC1.user_gates = {"noisy1": noisy_unitary1,
                "noisy2": noisy_unitary2,
                "noisy3": noisy_unitary3}

noise1 = np.random.normal(0, 0.01, 9)
noise1 = np.absolute(noise1)

QC1.add_gate("RX", targets=0, arg_value= 0.5)
QC1.add_gate("noisy1", targets=0, arg_value = noise1[0])
QC1.add_gate("noisy1", targets=1, arg_value = noise1[1])
QC1.add_gate("noisy1", targets=2, arg_value = noise1[2])
QC1.add_gate("RX", targets=1, arg_value= 0.1)
QC1.add_gate("RX", targets=2, arg_value= 0.2223472)
QC1.add_gate("CNOT", targets = 1, controls = 0)
QC1.add_gate("CNOT", targets = 2, controls = 0)
QC1.add_gate("noisy1", targets=0, arg_value = noise1[3])
QC1.add_gate("noisy1", targets=1, arg_value = noise1[4])
QC1.add_gate("noisy1", targets=2, arg_value = noise1[5])
QC1.add_gate("RX", targets = 0, arg_value = 0.26127)
QC1.add_gate("RX", targets = 1, arg_value= 1.3942948)
QC1.add_gate("RX", targets = 1, arg_value= 0.4378)
QC1.add_gate("noisy1", targets=0, arg_value = noise1[6])
QC1.add_gate("noisy1", targets=1, arg_value = noise1[7])
QC1.add_gate("noisy1", targets=2, arg_value = noise1[8])
U_list1 = QC1.propagators()
TestGate1 = gate_sequence_product(U_list1)
#print(TestGate)
print(calcGateFidelityN(TestGate1, TargetGate, 3))


QC2 = QubitCircuit(3) #testing noise used in poor statistics paper
QC2.user_gates = {"noisy1": noisy_unitary1,
                "noisy2": noisy_unitary2,
                "noisy3": noisy_unitary3}

noise2 = np.random.normal(0, 0.01, 9)
noise2 = np.absolute(noise2)

QC2.add_gate("RX", targets=0, arg_value= 0.5)
QC2.add_gate("noisy2", targets=0, arg_value = noise2[0])
QC2.add_gate("noisy2", targets=1, arg_value = noise2[1])
QC2.add_gate("noisy2", targets=2, arg_value = noise2[2])
QC2.add_gate("RX", targets=1, arg_value= 0.1)
QC2.add_gate("RX", targets=2, arg_value= 0.2223472)
QC2.add_gate("CNOT", targets = 1, controls = 0)
QC2.add_gate("CNOT", targets = 2, controls = 0)
QC2.add_gate("noisy2", targets=0, arg_value = noise2[3])
QC2.add_gate("noisy2", targets=1, arg_value = noise2[4])
QC2.add_gate("noisy2", targets=2, arg_value = noise2[5])
QC2.add_gate("RX", targets = 0, arg_value = 0.26127)
QC2.add_gate("RX", targets = 1, arg_value= 1.3942948)
QC2.add_gate("RX", targets = 1, arg_value= 0.4378)
QC2.add_gate("noisy2", targets=0, arg_value = noise2[6])
QC2.add_gate("noisy2", targets=1, arg_value = noise2[7])
QC2.add_gate("noisy2", targets=2, arg_value = noise2[8])
U_list2 = QC2.propagators()
TestGate2 = gate_sequence_product(U_list2)
#print(TestGate)
print(calcGateFidelityN(TestGate2, TargetGate, 3))

QC1 = QubitCircuit(3) #testing noise based on most general form
QC1.user_gates = {"noisy1": noisy_unitary1,
                "noisy2": noisy_unitary2,
                "noisy3": noisy_unitary3}

noise3 = np.random.normal(0, 0.01, 9)
noise3 = np.absolute(noise3)

QC1.add_gate("RX", targets=0, arg_value= 0.5)
QC1.add_gate("noisy3", targets=0, arg_value = noise3[0])
QC1.add_gate("noisy3", targets=1, arg_value = noise3[1])
QC1.add_gate("noisy3", targets=2, arg_value = noise3[2])
QC1.add_gate("RX", targets=1, arg_value= 0.1)
QC1.add_gate("RX", targets=2, arg_value= 0.2223472)
QC1.add_gate("CNOT", targets = 1, controls = 0)
QC1.add_gate("CNOT", targets = 2, controls = 0)
QC1.add_gate("noisy3", targets=0, arg_value = noise3[3])
QC1.add_gate("noisy3", targets=1, arg_value = noise3[4])
QC1.add_gate("noisy3", targets=2, arg_value = noise3[5])
QC1.add_gate("RX", targets = 0, arg_value = 0.26127)
QC1.add_gate("RX", targets = 1, arg_value= 1.3942948)
QC1.add_gate("RX", targets = 1, arg_value= 0.4378)
QC1.add_gate("noisy3", targets=0, arg_value = noise3[6])
QC1.add_gate("noisy3", targets=1, arg_value = noise3[7])
QC1.add_gate("noisy3", targets=2, arg_value = noise3[8])
U_list1 = QC1.propagators()
TestGate = gate_sequence_product(U_list1)
#print(TestGate)
print(calcGateFidelityN(TestGate, TargetGate, 3))

def N3qubitGateFunc(targetGate):
    def testGateParams(a1, a2, a3, a4, a5, a6):
        
        QC = QubitCircuit(3)
        QC.add_gate("RX", targets=0, arg_value= a1)
        QC.add_gate("RX", targets=1, arg_value= a2)
        QC.add_gate("RX", targets=2, arg_value= a3)
        QC.add_gate("CNOT", targets = 1, controls = 0)
        QC.add_gate("CNOT", targets = 2, controls = 0)
        QC.add_gate("RX", targets = 0, arg_value = a4)
        QC.add_gate("RX", targets = 1, arg_value= a5)
        QC.add_gate("RX", targets = 2, arg_value= a6)
        U_list = QC.propagators()
        finalGate = gate_sequence_product(U_list)
        return calcGateFidelityN(finalGate, targetGate, 3)
    return testGateParams

def N3qubitNoisyGateFunc(targetGate):
    def testGateParams(a1, a2, a3, a4, a5, a6):
        QC = QubitCircuit(3)

        noise = np.random.normal(0, 0.01, 6)
        noise = np.absolute(noise)

        QC.user_gates = {"N1": noisy_unitary1,
                        "N2": noisy_unitary2,
                        "N3": noisy_unitary3}
        QC.add_gate("RX", targets=0, arg_value= a1)
        QC.add_gate("RX", targets=1, arg_value= a2)
        QC.add_gate("RX", targets=2, arg_value= a3)
        QC.add_gate("noisy1", targets=0, arg_value = noise[0])
        QC.add_gate("noisy1", targets=1, arg_value = noise[1])
        QC.add_gate("noisy1", targets=2, arg_value = noise[2])
        QC.add_gate("CNOT", targets = 1, controls = 0)
        QC.add_gate("CNOT", targets = 2, controls = 0)
        QC.add_gate("RX", targets = 0, arg_value = a4)
        QC.add_gate("RX", targets = 1, arg_value= a5)
        QC.add_gate("RX", targets = 2, arg_value= a6)
        QC.add_gate("noisy1", targets=0, arg_value = noise[3])
        QC.add_gate("noisy1", targets=1, arg_value = noise[4])
        QC.add_gate("noisy1", targets=2, arg_value = noise[5])
        U_list = QC.propagators()
        finalGate = gate_sequence_product(U_list)
        return calcGateFidelityN(finalGate, targetGate, 3)
    return testGateParams