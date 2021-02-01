from problems import problem, hadamard
from qutip.qip.operations import snot
import qutip as qt

initialState = qt.basis(2, 0)

for _ in range(2):
    p = hadamard(initialState)
    p.default_opt()
    p.plot_result('test', True)
