from abc import ABC, abstractmethod

from itertools import product

import numpy as np
from scipy import stats
from scipy.special import factorial


class FisherInformation(ABC):

    def __init__(self, lamdas, Phi, sigma=1e-2, delta=1e-3):

        self.lamdas = np.array(lamdas)
        self.Phi = np.array(Phi)
        self.sigma = sigma
        self.delta = delta

        self.cube = get_high_prob_cube(lamdas, delta)

        self.I = None

    @abstractmethod
    def compute(self):

        if self.I is not None:
            return self.I
        
        pass

    def get_y_x_probs(self, ys, xs):

        M = self.Phi.shape[0]
        ps = 1 / (2 * np.pi * self.sigma ** 2) ** (M / 2) * np.exp(
            - 1 / 2 / self.sigma**2 * np.linalg.norm(
                ys[:, :, None] - (self.Phi @ xs)[:, None, :], axis=0
            )**2
        )
        return ps
    
    def get_x_probs(self, xs):

        ps = np.exp(-self.lamdas[:, None]) * (self.lamdas[:, None] ** xs) / factorial(xs, exact=True)
        return np.prod(ps, axis=0)


class FisherInformationNumericY1D(FisherInformation):

    def __init__(self, lamdas, Phi, n_points=100, **kwargs):

        super().__init__(lamdas, Phi, **kwargs)

        if not self.Phi.shape[0] == 1:
            raise ValueError('Measurements must be one-dimensional.')

        self.xs = np.indices([n + 1 for n in self.cube]).reshape((len(self.cube), -1))
        ys = self.Phi @ self.xs
        self.ys = np.linspace(np.min(ys) - 5 * self.sigma, np.max(ys) + 5 * self.sigma, n_points)[None, :]
        self.x_probs = self.get_x_probs(self.xs)
        self.y_x_probs = self.get_y_x_probs(self.ys, self.xs)
        self.y_probs = self.y_x_probs @ self.x_probs

    def compute(self):

        if self.I is not None:
            return self.I

        mats = np.einsum('ij,kl->ikjl', *([self.xs / self.lamdas[:, None] - 1] * 2))
        mats *= self.qq_int()[None, None, :, :]

        self.I = np.einsum('ijkl->ij', mats)
        return self.I

    def qq_int(self):

        integrand = self.y_x_probs.T[:, None, :] * self.y_x_probs.T[None, :, :]
        integrand *= self.x_probs[:, None, None] * self.x_probs[None, :, None]
        integrand /= self.y_probs[None, None, :]

        print(integrand.shape)
        return np.trapz(integrand, self.ys[None, :, :], axis=2)


class FisherInformationApprox(FisherInformation):

    def compute(self):

        if self.I is not None:
            return self.I

        xs = np.indices([n + 1 for n in self.cube]).reshape((len(self.cube), -1))
        n = xs.shape[-1]

        mats = np.einsum('ij,kl->ikjl', *([xs / self.lamdas[:, None] - 1] * 2))

        betas = np.zeros((n, n))
        for i in range(n):
            x = xs[:, i]
            for j in range(i + 1):
                x_prime = xs[:, j]
                betas[i, j] = self.beta(x, x_prime)
                betas[j, i] = betas[i, j]
        
        mats *= betas[None, None, :, :]
        probs = self.get_x_probs(xs)
        self.I = np.einsum('...ij,i,j->...', mats, probs, probs)
        return self.I

    def find_x_hat(self, y):

        xs = np.indices([n + 1 for n in self.cube]).reshape((len(self.cube), -1))
        ys = self.Phi @ xs
        dists = np.linalg.norm(y) ** 2 - 2 * y @ ys + np.linalg.norm(ys, axis=0) ** 2
        i = np.argmin(dists)
        return xs[:, i]
    
    def beta(self, x, x_prime):

        x_avg = (x + x_prime) / 2
        x_diff = (x - x_prime) / 2
        x_hat = self.find_x_hat(self.Phi @ x_avg)
        return np.exp(
            1 / self.sigma**2 * (
                np.linalg.norm(self.Phi @ (x_hat - x_avg)) ** 2
                - np.linalg.norm(self.Phi @ x_diff) ** 2
            ) 
        ) / self.get_x_probs(x_hat[:, None])[0]


def get_high_prob_cube(lamdas, delta=1e-3):

    return [int(stats.poisson.ppf(1 - delta, lamda)) for lamda in lamdas]

