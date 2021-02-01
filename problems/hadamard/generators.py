import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def calcStateFidelity(finalState, targetState):
    finalState = finalState.dag()
    return abs(targetState.overlap(finalState))**2

def calcGateFidelity(finalGate, targetGate):
    product = finalGate * targetGate.dag()
    return (abs(product[0][0][0] + product[1][0][1])**2) / 4

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

def generatePropagator(a1, a2):
    R_z = (qt.sigmaz() * 1j * a1).expm()
    R_y = (qt.sigmay() * 1j * a2).expm()
    return (-1j) * R_z * R_y

