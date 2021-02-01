from ..problem import problem
from .generators import *
from qutip.qip.operations import snot
import qutip as qt


class hadamard(problem):
    def __init__(self, initialState, targetGate=snot(1), configPath='./problems/hadamard/config.yaml'):
        targetState = targetGate * initialState
        testState = singleStateFunc(initialState, targetState)
        testGate = singleGateFunc(targetGate)
        problem.__init__(self, testState=testState, testGate=testGate, configPath=configPath)