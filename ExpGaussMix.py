import numpy as np
import numpy.linalg as la
from copy import deepcopy

# Import external dependencies
import GaussModel as Gauss
import ExpGauss as ExpG
from LocationScaleProbability import LSPD


class Model:
    # Initialize the standard exponential modified Gaussian mixture model.
    def __init__(self, a=1, z=0, optA=False, optZ=False):
        self.ExpG = ExpG.Model(a, optA)
        self.Gauss = Gauss.Model()
        self.z = z
        self.optZ = optZ

        # Retrieve the current value of the indicator parameter z.

    def GetZ(self):
        return self.z

    # Update the indicator parameter z.
    def SetZ(self, z):
        self.z = z
        return self.z

    # Retrieve parameters a and z.
    def GetAZ(self):
        self.a = self.ExpG.GetA()
        return self.a, self.z

    # Set parameters a and z.
    def SetAZ(self, a, z):
        self.ExpG.SetA(a)
        self.z = z

    # Set optimization flags for parameters a and z.
    def SetOpt(self, optA, optZ):
        self.ExpG.optA = optA
        self.optZ = optZ

    # Verify if the model parameters are valid.
    def IsValid(self):
        if self.ExpG.IsValid() and self.Gauss.IsValid():
            return True
        else:
            return False

    # Force parameters into a valid state.
    def MakeValid(self, thres=1e-6):
        self.ExpG.MakeValid()
        self.Gauss.MakeValid()
        return self

    # Copy variables from another object to the current one.
    def Assign(self, other):
        self.ExpG.Assign(other.ExpG)
        self.z = other.z

    # Add two model instances together.
    def __add__(self, other):
        return Model(self.ExpG.GetA() + other.ExpG.GetA(), self.z + other.z, self.ExpG.GetOptA(), self.optZ)

    # Subtract one model instance from another.
    def __sub__(self, other):
        return Model(self.ExpG.GetA() - other.ExpG.GetA(), self.z - other.z, self.ExpG.GetOptA(), self.optZ)

    # Scale the model by a scalar value.
    def __mul__(self, scalar):
        return Model(self.ExpG.GetA() * scalar, self.z * scalar, self.ExpG.GetOptA(), self.optZ)

    # Divide one model instance by another.
    def __truediv__(self, other):
        return Model(self.ExpG.GetA() / other.ExpG.GetA(), self.z / other.z, self.ExpG.GetOptA(), self.optZ)

    # Calculate the norm of the model parameters.
    def norm(self):
        return np.sqrt(self.ExpG.norm() ** 2 + la.norm(self.z) ** 2)

    # Compute the negative log-likelihood function of the ExpGaussMix distribution.
    def NegLogDen(self, x):
        z = self.GetZ()
        return (1 - z) * self.Gauss.NegLogDen(x) + z * self.ExpG.NegLogDen(x)

    # Calculate the density function for the given input.
    def Density(self, x):
        return np.exp(-self.NegLogDen(x))

    # Compute the first gradient with respect to x.
    def GradX(self, x):
        z = self.GetZ()
        return (1 - z) * self.Gauss.GradX(x) + z * self.ExpG.GradX(x)

    # Compute the second gradient with respect to x.
    def GradX2(self, x):
        z = self.GetZ()
        return (1 - z) * self.Gauss.GradX2(x) + z * self.ExpG.GradX2(x)

    # Compute the gradient with respect to parameter a.
    def GradA(self, x):
        z = self.GetZ()
        return z * self.ExpG.GradA(x)

    # Compute the second gradient with respect to parameter a.
    def GradA2(self, x):
        z = self.GetZ()
        return z * self.ExpG.GradA2(x)

    # Return a model containing the gradient with respect to parameter a.
    def Gradient(self, x):
        return Model(np.sum(self.GradA(x)) if self.ExpG.GetOptA() else 0, 0)

    # Compute the Laplacian of the model.
    def Laplacian(self, x):
        return Model(np.sum(self.GradA2(x)) if self.ExpG.GetOptA() else 0, 0)

    # Calculate a scaled gradient for optimization stability.
    def ScaledGradient(self, x, d=1e-12):
        return Model(np.sum(self.GradA(x)) / (abs(np.sum(self.GradA2(x)) + d)) if self.ExpG.GetOptA() else 0, 0)

    # Generate samples from the mixture distribution.
    def genSamples(self, size=1):
        z = self.GetZ()
        ind = np.random.random(size) < z
        return (1 - ind) * self.Gauss.GenSamples(size) + ind * self.ExpG.GenSamples(size)

    # Calculate the expected value of z given the mixture probabilities.
    def ExpectedZ(self, x, mix):
        GauDen = self.Gauss.Density(x)
        ExpGDen = self.ExpG.Density(x)

        ind = (mix * ExpGDen + (1 - mix) * GauDen) == 0
        notInd = np.logical_not(ind)
        z = np.zeros(x.shape)
        z[notInd] = (mix * ExpGDen[notInd]) / (mix * ExpGDen[notInd] + (1 - mix) * GauDen[notInd])
        z[np.logical_and(ind, x > 0)] = 1
        z[np.logical_and(ind, x < 0)] = 0
        return z


class ExpGaussMix(LSPD):
    # Initialize the exponential modified Gaussian mixture model with location and scale parameters.
    def __init__(self, a=1.0, m=0, s=1, z=0.0, optA=False, optM=False, optS=False, optZ=False):

        super().__init__(m, s, optM, optS)
        self.std = Model(a, z, optA, optZ)
        self.m = m
        self.s = s
        self.optM = optM
        self.optS = optS

    # Retrieve parameters a, m, s, and z.
    def GetAMSZ(self):
        a, z = self.std.GetAZ()
        return a, self.GetMS(), z

    # Set model parameters a, m, s, and z.
    def SetAMSZ(self, a, m, s, z):
        self.std.SetAZ(a, z)
        self.SetMS(m, s)

    # Retrieve the indicator parameter z.
    def GetZ(self):
        return self.std.GetZ()

    # Check if parameter z is set to be optimized.
    def GetOptZ(self):
        return self.std.optZ

    # Set the indicator parameter z.
    def SetZ(self, z):
        return self.std.SetZ(z)

    # Set optimization flags for all model parameters.
    def SetOpt(self, optA, optM, optS, optZ):
        self.std.SetOpt(optA, optZ)
        self.SetOptM(optM)
        self.SetOptS(optS)

    # Calculate the mean of the indicator parameter z.
    def CalculateMix(self):
        return np.mean(self.GetZ())

    # Calculate the expected value of z given mixture probability.
    def ExpectedZ(self, x, mix):
        m, s = self.GetMS()
        self.std.ExpectedZ((x - m) / s, mix)
        return self.std.ExpectedZ((x - m) / s, mix)

    # Perform an expectation step to update the indicator parameter z.
    def ExpectationStep(self, x, mix):
        self.SetZ(self.GetOptZ() * self.ExpectedZ(x, mix))
        return self

    # Perform a maximization step to update continuous parameters and mixture coefficient.
    def MaximizationStep(self, x, mix, optMix, maxIter):
        super(ExpGaussMix, self).Optimize(x, maxIter=maxIter)
        if optMix:
            mix = np.mean(self.GetZ())
        return mix

    # Optimize the model parameters using the EM algorithm.
    def Optimize(self, x, mix=0.5, optMix=True, maxIter=8, minChange=1e-6, maxMaxIter=128):
        converged = False
        iter = 0
        while not converged:
            oldSelf = deepcopy(self)
            mix = self.MaximizationStep(x, mix, optMix, maxMaxIter)

            if np.any(self.GetOptZ()):
                self.ExpectationStep(x, mix)

            iter += 1
            if iter > maxIter or (oldSelf - self).norm(la) < minChange:
                converged = True
        return self

    def getZ(self):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load dataset
    file = r"C:\改正基线校正CPU的版本\Protein_DSC_Data\Protein_DSC_Data/combative.csv"
    df = pd.read_csv(file, header=None)
    df = df.T
    Q = df.iloc[0, 1:]
    Q = Q.astype(float)
    Q = np.array(Q, dtype=np.float32)
    I = df.iloc[1:, 1:]
    I = I.astype(float)
    I = np.array(I, dtype=np.float32)
    x = -I
    std = 0.7
    mix = 0.5

    ParVariable = ExpGaussMix(std, z=mix)
    dom = np.zeros(x.shape)
    GaussIsIn = np.ones(x.shape)

    for j in range(len(x)):
        ParVariable.SetOpt(True, True, True, True)
        ParVariable.Optimize(x[j], mix)
        ParChange = ParVariable.GetZ()
        print(ParChange)
        for k in range(len(ParChange)):
            if ParChange[k] < mix - 0.007:
                dom[j, k] = 1

    # Visualization of the sample probability signal distribution
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(Q, x[j], c='C1', label='Measuring signal')
    ax2 = ax1.twinx()
    ax2.plot(Q, dom[j], c='C0', label='Distribution probability')
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc='best')
    ax1.set_xlabel("Temperature", size=18)
    ax1.set_ylabel("Heat capacity", size=18)
    ax2.set_ylabel("Distribution probability", size=18)
    plt.title(f"scan={j}")
    plt.show()