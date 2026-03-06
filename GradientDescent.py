import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


###### Gradient descent ######

# Defines and returns a dictionary of hyperparameters used for the optimization algorithms.
def DefineOptimizationParameters(minDecrease=1e-8, maxIter=1024, minIter=0,
                                 maxDampingIter=32, dampingFactor=7, increaseFactor=2,
                                 initialStepSize=1):
    params = {'dampingFactor': dampingFactor, 'increaseFactor': increaseFactor,
              'minDecrease': minDecrease, 'initialStepSize': initialStepSize,
              'maxIter': maxIter, 'minIter': minIter, 'maxDampingIter': maxDampingIter}
    return params


# Performs gradient descent optimization with an adaptive step size and damping mechanism.
def GradientDescent(X, objective, gradient,
                    projection=(lambda x: x),
                    updateVariables=(lambda x, dx, s: x - s * dx),
                    params=DefineOptimizationParameters(),
                    isInDomain=lambda x: True):
    X = projection(X)

    obj = objective(X)

    stepSize = params['initialStepSize']
    converged = False
    iter = 0
    objArr = np.zeros(params['maxIter'] + 1)
    stepArr = np.zeros(params['maxIter'] + 1)
    XTemp = deepcopy(X)

    while not converged:
        oldObj = obj

        deltaX = gradient(X)
        XTemp = updateVariables(X, deltaX, stepSize)
        XTemp = projection(XTemp)
        obj = objective(XTemp)

        dampingIter = 0
        while (obj > oldObj and dampingIter < params['maxDampingIter']) or not isInDomain(X):
            stepSize /= params['dampingFactor']

            XTemp = updateVariables(X, deltaX, stepSize)
            XTemp = projection(XTemp)
            obj = objective(XTemp)
            dampingIter += 1

        stepSize *= params['increaseFactor']
        X = deepcopy(XTemp)
        objArr[iter] = obj
        stepArr[iter] = stepSize
        iter += 1

        if (iter > params['maxIter'] or oldObj - params['minDecrease'] < obj) and iter > params['minIter']:
            converged = True

    return X, objArr[0:iter], stepArr[0:iter]


# Executes a Gauss-Newton optimization method using adaptive step sizing and damping.
def GaussNewton(X, objective, gradient,
                projection=(lambda x: x),
                updateVariables=(lambda x, dx, s: x - s * dx),
                params=DefineOptimizationParameters(),
                isInDomain=lambda x: True):
    X = projection(X)
    obj = objective(X)
    stepSize = params['initialStepSize']
    converged = False
    iter = 0
    objArr = np.zeros(params['maxIter'] + 1)
    stepArr = np.zeros(params['maxIter'] + 1)
    XTemp = deepcopy(X)

    while not converged:
        oldObj = obj

        deltaX = gradient(X)
        XTemp = updateVariables(X, deltaX, stepSize)
        XTemp = projection(XTemp)
        obj = objective(XTemp)

        dampingIter = 0
        while (obj > oldObj and dampingIter < params['maxDampingIter']) or not isInDomain(X):
            stepSize /= params['dampingFactor']

            XTemp = updateVariables(X, deltaX, stepSize)
            XTemp = projection(XTemp)
            obj = objective(XTemp)
            dampingIter += 1

        stepSize *= params['increaseFactor']
        X = deepcopy(XTemp)
        objArr[iter] = obj
        stepArr[iter] = stepSize
        iter += 1

        if (iter > params['maxIter'] or oldObj - params['minDecrease'] < obj) and iter > params['minIter']:
            converged = True

    return X, objArr[0:iter], stepArr[0:iter]


# Performs gradient descent optimization using a fixed step size without a damping loop.
def GradientDescentFixedStep(X, objective, gradient,
                             projection=(lambda x: x),
                             updateVariables=(lambda x, dx, s: x - s * dx),
                             params=DefineOptimizationParameters(),
                             isInDomain=lambda x: True):
    X = projection(X)
    obj = objective(X)
    stepSize = params['initialStepSize']
    converged = False
    iter = 0
    objArr = np.zeros(params['maxIter'] + 1)
    stepArr = np.zeros(params['maxIter'] + 1)
    XTemp = deepcopy(X)

    while not converged:
        oldObj = obj

        deltaX = gradient(X)
        XTemp = updateVariables(X, deltaX, stepSize)
        XTemp = projection(XTemp)
        obj = objective(XTemp)

        X = deepcopy(XTemp)
        stepArr[iter] = stepSize
        objArr[iter] = obj
        iter += 1

        if (iter > params['maxIter'] or oldObj - params['minDecrease'] < obj) and iter > params['minIter']:
            converged = True

    return X, objArr[0:iter], stepArr[0:iter]


# Implements the Expectation-Maximization (EM) algorithm to iteratively update parameters until convergence.
def EM(p, expectationStep, maximizationStep, maxIter=32, minDecrease=1e-6):
    converged = False
    iter = 0
    while not converged:

        p = expectationStep(p)

        p = maximizationStep(p)

        iter += 1

        if np.linalg.norm(Z - oldZ) < minDecrease or iter > maxIter:
            converged = True

    return T, Z, iter