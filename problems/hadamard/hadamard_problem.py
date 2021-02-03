from ..problem import problem
from .hadamard_generators import *
from qutip.qip.operations import snot
import qutip as qt

class hadamard(problem):
    def __init__(self, initialState, targetGate=snot(1), configPath='./problems/hadamard/hadamard_config.yaml', verbose=2):
        targetState = targetGate * initialState
        testState_list = [singleStateFunc(initialState, targetState)]
        testGate = singleGateFunc(targetGate)
        problem.__init__(self, testState_list=testState_list, testGate=testGate, configPath=configPath, verbose=verbose)
