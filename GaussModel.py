import numpy as np

from LocationScaleProbability import LSPD

###### Standard Gaussian model ######

# Represents a standard Gaussian model with basic operations and likelihood functions.
class Model():

    # Computes the negative log likelihood function.
    def NegLogDen(self, x):
        return x**2/2 + np.log(2*np.pi) / 2

    # Computes the first derivative of the negative log likelihood function.
    def GradX(self, x):
        return x

    # Computes the second derivative of the negative log likelihood function.
    def GradX2(self, x):
        return np.ones(x.shape)

    # Returns the current instance as the gradient.
    def Gradient(self, x):
        return self

    # Returns the current instance as the Laplacian.
    def Laplacian(self, x):
        return self

    # Returns the current instance as the scaled gradient.
    def ScaledGradient(self, x, d = 1e-12):
        return self

    # Prints nothing for the base model.
    def print(self):
        return None

    # Checks if the model parameters are valid.
    def IsValid(self):
        return True

    # Returns the instance to maintain validity.
    def MakeValid(self):
        return self

    # Assigns properties from another object.
    def Assign(self, other):
        return None

    # Generates samples from a standard normal distribution.
    def GenSamples(self, size = 1):
        return np.random.standard_normal(size)

    # Defines the addition operator for the standard normal distribution.
    def __add__(self, other):
        return self

    # Defines the subtraction operator for the standard normal distribution.
    def __sub__(self, other):
        return self

    # Defines the multiplication operator for the standard normal distribution.
    def __mul__(self, scalar):
        return self

    # Defines the division operator for the standard normal distribution.
    def __truediv__(self, other):
        return self

    # Returns the norm of the model.
    def norm(self):
        return 0

    # Computes the standard normal distribution probability density function.
    def Density(self, x):
        return 1/(np.sqrt(2*np.pi)) * np.exp( - x**2 / 2 )

    # Computes the first-order derivative of the likelihood function.
    def DenGradX(self, x):
        return -x * self.Density(x)

    # Computes the second-order derivative of the likelihood function.
    def DenGradX2(self, x):
        return - self.Density(x) + (-x * self.DenGradX(x))

###### Gaussian Distribution ######

# Represents a Gaussian distribution extending the location-scale probability distribution.
class Gauss( LSPD ):

    # Initializes the Gaussian distribution with a standard model, location, scale, and optimization flags.
    def __init__(self, m = 0, s = 1, optM = False, optS = False):
        self.std = Model()
        self.m = m
        self.s = s

        self.optM = optM
        self.optS = optS