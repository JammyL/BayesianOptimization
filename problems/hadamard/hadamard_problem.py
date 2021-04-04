from ..problem import problem
from .hadamard_generators import *
from qutip.qip.operations import snot
import qutip as qt

class hadamard(problem):
    def __init__(self, initialState_list, targetGate=snot(1), epsilon=0, configPath='./problems/hadamard/hadamard_config.yaml', verbose=2):
        targetState_list = [targetGate * initialState for initialState in initialState_list]
        testState_list = [singleStateFunc(initialState_list[i], targetState_list[i], epsilon) for i in range(len(initialState_list))]
        testGate = singleGateFunc(targetGate, epsilon)
        problem.__init__(self, testState_list=testState_list, testGate=testGate, configPath=configPath, verbose=verbose)
