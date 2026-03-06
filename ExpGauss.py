import numpy as np
import numpy.linalg as la
from scipy.stats import (skew, moment, norm)
from scipy.special import erfcx, erfc
import GradientDescent as GD
from LocationScaleProbability import LSPD
from SpecialFunctions import log_erfc


class Model:

    # Initialize the standard exponential modified Gaussian model with parameter a and its optimization flag.
    def __init__(self, a=1, optA=False):
        self.a = a
        self.optA = optA

    # Set the shape parameter a.
    def SetA(self, a):
        self.a = a

    # Get the current shape parameter a.
    def GetA(self):
        return self.a

    # Set the optimization flag for parameter a.
    def SetOptA(self, optA):
        self.optA = optA

    # Get the current optimization flag for parameter a.
    def GetOptA(self):
        return self.optA

    # Override the addition operator to return a new Model instance with summed parameters.
    def __add__(self, other):
        return Model(self.a + other.a, self.optA)

    # Override the subtraction operator to return a new Model instance with subtracted parameters.
    def __sub__(self, other):
        return Model(self.a - other.a, self.optA)

    # Override the multiplication operator to scale parameter a by a scalar.
    def __mul__(self, scalar):
        return Model(self.a * scalar, self.optA)

    # Override the division operator to return a new Model instance with divided parameters.
    def __truediv__(self, other):
        return Model(self.a / other.a, self.optA)

    # Calculate the norm of parameter a.
    def norm(self):
        return la.norm(self.a)

    # Display the current parameter values and their optimization status.
    def print(self):
        print('a = ' + str(self.a))
        print('optA = ' + str(self.optA))

    # Verify if parameter a is in a valid state.
    def IsValid(self):
        return True if self.a > 0 else False

    # Force parameter a to be within a valid range.
    def MakeValid(self, thres=1e-6):
        self.a = max(self.a, int(thres))

    # Copy the parameter a from another model object to this one.
    def Assign(self, other):
        self.a = other.a

    # Generate samples from the standard exponential modified Gaussian distribution.
    def GenSamples(self, size=1):
        return np.random.standard_normal(size) + np.random.exponential(scale=1 / self.a, size=size)

    # Compute the negative log-likelihood of the model for given input x.
    def NegLogDen(self, x):
        a = self.a
        d = (a - x) / np.sqrt(2)
        safe_d = np.clip(d, -700, 700)
        nld = -np.log(a / 2) - a ** 2 / 2 + a * x - log_erfc(safe_d)
        nld = np.where(np.isinf(nld), 1e10, nld)
        return nld

        # Calculate the probability density function value for the given input x.

    def Density(self, x):
        return np.exp(-self.NegLogDen(x))

    # Check if the function is convex with respect to parameter a.
    def IsConvexInA(self):
        return (True if self.a < 1 else False)

    # Helper method to calculate the intermediate variable d.
    def _get_d(self, x):
        a = self.a
        d = (a - x) / np.sqrt(2)
        return d

    # Helper method to calculate intermediate variables d and e.
    def _get_de(self, x):
        d = self._get_d(x)
        e = 1 / erfcx(d)
        return d, e

    # Compute the first derivative of the negative log density with respect to x.
    def GradX(self, x):
        a = self.a
        d, e = self._get_de(x)
        return a - np.sqrt(2 / np.pi) * e

    # Compute the second derivative of the negative log density with respect to x.
    def GradX2(self, x):
        a = self.a
        d, e = self._get_de(x)
        return (2 / np.pi * e ** 2 - 2 / np.sqrt(np.pi) * e * d)

    # Compute the first derivative of the negative log density with respect to parameter a.
    def GradA(self, x):
        a = self.a
        d, e = self._get_de(x)
        return -(1 / a + a) + x + np.sqrt(2 / np.pi) * e

    # Compute the second derivative of the negative log density with respect to parameter a.
    def GradA2(self, x):
        a = self.a
        d, e = self._get_de(x)
        return (1 / a ** 2 - 1) + 2 / np.pi * e ** 2 - 2 / np.sqrt(np.pi) * e * d

    # Calculate the gradient for parameter a.
    def Gradient(self, x):
        return Model(np.sum(self.GradA(x)) if self.optA else 0)

    # Calculate the Laplacian of the model with respect to parameter a.
    def Laplacian(self, x):
        return Model(np.sum(self.GradA2(x)) if self.optA else 0)

    # Calculate the scaled gradient for parameter a to assist in optimization stability.
    def ScaledGradient(self, x, d=1e-12):
        return Model(np.sum(self.GradA(x)) / (abs(np.sum(self.GradA2(x)) + d)) if self.optA else 0)


class ExpG(LSPD):

    # Initialize the exponential modified Gaussian model with shape, location, and scale parameters.
    def __init__(self, a=1, m=0, s=1, optA=False, optM=False, optS=False):
        self.m = m
        self.s = s
        self.std = Model(a, optA)
        self.optM = optM
        self.optS = optS

    # Return a tuple containing the shape parameter a and the location-scale parameters.
    def GetAMS(self):
        return self.std.a, self.GetMS()

    # Set the shape, location, and scale parameters of the model.
    def SetAMS(self, a, m, s):
        self.std.SetA(a)
        self.m = m
        self.s = s

    # Set the shape parameter a.
    def SetA(self, a):
        self.std.SetA(a)

    # Set the optimization flag for the shape parameter a.
    def SetOptA(self, optA):
        self.std.SetOptA(optA)

    # Calculate the gradient of the log-likelihood with respect to parameter a.
    def GradA(self, x):
        m, s = self.GetMS()
        return self.std.GradA((x - m) / s)

    # Calculate the second derivative of the log-likelihood with respect to parameter a.
    def GradA2(self, x):
        m, s = self.GetMS()
        return self.std.GradA2((x - m) / s)


if __name__ == '__main__':

    # Create an ExpG object and display its initial parameters.
    E = ExpG()
    E.print()

    # Generate random samples from the model.
    n = 1024
    x = E.GenSamples(n)

    # Set specific parameters and print them.
    E.SetAMS(.5, 0, 1)
    E.print()

    # Configure optimization flags for parameters.
    E.SetOptA(True)
    E.SetOptM(False)
    E.SetOptS(False)

    # Print the model state before optimization.
    E.print()

    # Optimize the model parameters based on the generated data.
    E.Optimize(x)

    # Print the model state after optimization.
    E.print()