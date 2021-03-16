from ..problem import problem
import numpy as np

class sinusoid(problem):
    def __init__(self, initialState_list, targetGate=lambda x: np.sin(x)**2, configPath='./problems/hadamard/hadamard_config.yaml', verbose=2):
        problem.__init__(self, testState_list=[lambda x: np.sin(x)**2], testGate=lambda x: -np.sin(x),
                        configPath=configPath, verbose=verbose)
