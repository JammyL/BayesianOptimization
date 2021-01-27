import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def calcStateInfidelity(finalState, targetState):
    finalState = finalState.dag()
    return abs(targetState.overlap(finalState))**2

def calcOpInfidelity(finalOp, targetOp):
    product = finalOp * targetOp.dag()
    # if  abs(qt.metrics.hilbert_dist(product, qt.qeye(2))) < 1e-3:
    #     print(product)
    # return 1 - abs(qt.metrics.hilbert_dist(product, qt.qeye(2)))
    return (abs(product[0][0][0] + product[1][0][1])**2) / 4

def singleStateFunc(initialState, targetState):
    def testStateParams(a1, a2):
        R_z = (qt.sigmaz() * 1j * a1).expm()
        R_y = (qt.sigmay() * 1j * a2).expm()
        propagator = (-1j) * R_z * R_y
        propagatedState = propagator * initialState
        return calcStateInfidelity(propagatedState, targetState)
    return testStateParams

def singleOpFunc(targetOp):
    def testOperatorParams(a1, a2):
        R_z = (qt.sigmaz() * 1j * a1).expm()
        R_y = (qt.sigmay() * 1j * a2).expm()
        propagator = (-1j) * R_z * R_y
        return calcOpInfidelity(propagator, targetOp)
    return testOperatorParams

def generatePropagator(a1, a2):
    R_z = (qt.sigmaz() * 1j * a1).expm()
    R_y = (qt.sigmay() * 1j * a2).expm()
    return (-1j) * R_z * R_y


def plot_bo(bo, title):
    X, Y = np.mgrid[0:2:1000j, -2:2:1000j]
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    mean, sigma = bo._gp.predict(positions, return_std=True)
    mean = np.reshape(mean, (-1,1000))
    sigma = np.reshape(sigma, (-1, 1000))
    mean = 1 - mean

    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    c1 = ax1.contourf(X,Y, mean, cmap='bwr', levels = 20)
    c2 = ax2.contourf(X,Y, sigma, cmap='bwr', levels = 20)
    plt.colorbar(c1, ax = ax1)
    plt.colorbar(c2, ax = ax2)

    ax1.set_xlabel('a1')
    ax1.set_ylabel('a2')
    ax2.set_xlabel('a1')
    ax2.set_ylabel('a2')

    ax1.set_title(title + ': Mean')
    ax2.set_title(title + ': Sigma')

    # fig1.savefig("plus_op_mean.png")
    # fig2.savefig("plus_op_sigma.png")

