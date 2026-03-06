import numpy as np
import numpy.linalg as plt
import GradientDescent as GD


###### Standard location scale probability distribution ######

# Represents a standard location-scale probability distribution.
class LSPD:

    # Initializes the distribution with a standard distribution, location, scale, and optimization flags.
    def __init__(self, std, m=0, s=1, optM=False, optS=False):
        self.std = std
        self.m = m
        self.s = s
        self.optM = optM
        self.optS = optS

    # Prints the current values of the location, scale, and their respective optimization flags.
    def print(self):
        print('m = ' + str(self.m))
        print('optM =' + str(self.optM))
        print('s = ' + str(self.s))
        print('optS =' + str(self.optS))
        self.std.print()

    # Checks if the current probability distribution parameters are mathematically valid.
    def IsValid(self):
        if np.all(self.s > 0) and self.std.IsValid():
            return True
        else:
            return False

    # Ensures the scale parameter remains above a specified threshold to maintain validity.
    def MakeValid(self, thres=1e-6):
        self.std.MakeValid()
        self.s = np.maximum(thres, self.s)
        return self

    # Assigns the variables from another LSPD object to the current instance.
    def Assign(self, other):
        self.std.Assign(other.std)
        self.m = other.m
        self.s = other.s

    # Defines the addition operator for the location-scale distribution family.
    def __add__(self, other):
        return LSPD(self.std + other.std, self.m + other.m, self.s + other.s, self.optM, self.optS)

    # Defines the subtraction operator for the location-scale distribution family.
    def __sub__(self, other):
        return LSPD(self.std - other.std, self.m - other.m, self.s - other.s, self.optM, self.optS)

    # Defines the scalar multiplication operator for the location-scale distribution family.
    def __mul__(self, scalar):
        return LSPD(self.std * scalar, self.m * scalar, self.s * scalar, self.optM, self.optS)

    # Defines the true division operator for the location-scale distribution family.
    def __truediv__(self, other):
        return LSPD(self.std / other.std, self.m / other.m, self.s / other.s, self.optM, self.optS)

    # Computes the norm of the distribution parameters.
    def norm(self, la):
        return np.sqrt(self.std.norm() ** 2 + la.norm(self.m) ** 2 + la.norm(self.s) ** 2)

    # Returns the location and scale parameters.
    def GetMS(self):
        return self.m, self.s

    # Sets both the location and scale parameters.
    def SetMS(self, m, s):
        self.m = m
        self.s = s

    # Sets the location parameter.
    def SetM(self, m):
        self.m = m

    # Returns the location parameter.
    def GetM(self):
        return self.m

    # Sets the scale parameter.
    def SetS(self, s):
        self.s = s

    # Returns the scale parameter.
    def GetS(self):
        return self.s

    # Sets the optimization flag for the location parameter.
    def SetOptM(self, optM):
        self.optM = optM

    # Sets the optimization flag for the scale parameter.
    def SetOptS(self, optS):
        self.optS = optS

    # Sets optimization flags for multiple parameters simultaneously.
    def SetOpt(self, optA, optM, optS, optZ):
        self.std.SetOpt(optA, optZ)
        self.SetOptM(optM)
        self.SetOptS(optS)

    # Generates samples based on the current location and scale parameters.
    def GenSamples(self, size=1):
        if size > 1:
            return self.s * self.std.GenSamples(size) + self.m
        elif size == 1:
            return self.s * self.std.GenSamples(self.m) + self.m

    # Computes the negative logarithmic density function.
    def NegLogDen(self, x):
        m, s = self.GetMS()
        return self.std.NegLogDen((x - m) / s) + np.log(s)

    # Computes the first-order derivative of the probability density function with respect to the location parameter.
    def GradM(self, x):
        m, s = self.GetMS()
        return -1 / s * self.std.GradX((x - m) / s)

    # Computes the second-order derivative of the probability density function with respect to the location parameter.
    def GradM2(self, x):
        m, s = self.GetMS()
        return 1 / s ** 2 * self.std.GradX2((x - m) / s)

    # Computes the first-order derivative of the probability density function with respect to the scale parameter.
    def GradS(self, x):
        m, s = self.GetMS()
        xm = (x - m) / s
        return self.std.GradX(xm) * -xm / s + 1 / s

    # Computes the second-order derivative of the probability density function with respect to the scale parameter.
    def GradS2(self, x):
        m, s = self.GetMS()
        xm = (x - m) / s
        return (self.std.GradX2(xm) * (xm / s) ** 2 + self.std.GradX(xm) * 2 * xm / s ** 2) - 1 / s ** 2

    # Computes the first-order derivative of the probability density function with respect to x.
    def GradX(self, x):
        m, s = self.GetMS()
        return 1 / s * self.std.GradX((x - m) / s)

    # Computes the second-order derivative of the probability density function with respect to x.
    def GradX2(self, x):
        m, s = self.GetMS()
        return 1 / s ** 2 * self.std.GradX2((x - m) / s)

        # Calculates the aggregated gradient based on active optimization flags.

    def Gradient(self, x):
        gradM = np.sum(self.GradM(x)) if self.optM else 0
        gradS = np.sum(self.GradS(x)) if self.optS else 0
        return LSPD(self.std.Gradient(x), gradM, gradS, self.optM, self.optS)

    # Calculates the aggregated Laplacian based on active optimization flags.
    def Laplacian(self, x):
        gradM2 = np.sum(self.GradM2(x)) if self.optM else 0
        gradS2 = np.sum(self.GradS2(x)) if self.optS else 0
        return LSPD(self.std.Laplacian(x), gradM2, gradS2, self.optM, self.optS)

    # Computes the scaled gradient for optimization, applying scaling in the log domain.
    def ScaledGradient(self, x, d=1e-12):
        gradM = np.sum(self.GradM(x)) / np.sum(self.GradM2(x)) if self.optM else 0
        gradS = np.sum(self.GradS(x)) / (abs(np.sum(self.GradS2(x))) + d) if self.optS else 0
        return LSPD(self.std.ScaledGradient(x), gradM, gradS, self.optM, self.optS)

    # Calculates the total negative log-likelihood for a given input array.
    def NegLogLike(self, x):
        return np.sum(self.NegLogDen(x))

    # Performs parameter optimization using gradient descent based on the provided data.
    def Optimize(self, x, maxIter=32, plot=False):
        params = GD.DefineOptimizationParameters(maxIter=maxIter, minDecrease=1e-5)
        obj = lambda E: E.NegLogLike(x)
        grad = lambda E: E.ScaledGradient(x)
        updateVariables = lambda E, dE, s: E - (dE * s)
        projection = lambda E: E.MakeValid()
        E, normArr, stepArr = GD.GradientDescent(self, obj, grad, projection, updateVariables, params)
        self.Assign(E)
        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.plot(normArr)
            plt.subplot(122)
            plt.plot(stepArr)
            plt.show()
        return self

    # Computes the probability density for a given input.
    def Density(self, x):
        return np.exp(-self.NegLogDen(x))

    # Computes the gradient of the density using the density and negative log-density gradient.
    def DenGrad(self, den, nllGrad):
        return den * -nllGrad

    # Computes the second-order gradient of the density using negative log-density gradients.
    def DenGrad2(self, den, nllGrad, nllGrad2):
        return den * (nllGrad ** 2 - nllGrad2)

    # Computes the density gradient with respect to x.
    def DenGradX(self, x):
        return self.DenGrad(self.Density(x), self.GradX(x))

    # Computes the second-order density gradient with respect to x.
    def DenGradX2(self, x):
        return self.DenGrad2(self.Density(x), self.GradX(x), self.GradX2(x))

    # Computes the density gradient with respect to the location parameter.
    def DenGradM(self, x):
        return self.DenGrad(self.Density(x), self.GradM(x))

    # Computes the second-order density gradient with respect to the location parameter.
    def DenGradM2(self, x):
        return self.DenGrad2(self.Density(x), self.GradM(x), self.GradM2(x))

    # Computes the density gradient with respect to the scale parameter.
    def DenGradS(self, x):
        return self.DenGrad(self.Density(x), self.GradS(x))

    # Computes the second-order density gradient with respect to the scale parameter.
    def DenGradS2(self, x):
        return self.DenGrad2(self.Density(x), self.GradS(x), self.GradS2(x))