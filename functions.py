#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from scipy.stats import kendalltau
from numba import njit, jit


@njit(fastmath=True)
def integrate_model(
    Wij: np.array,
    Tmat: np.array,
    rbeta: np.array,
    dt: float,
    beta: float,
    tmax: float = 8,
):
    Npop: int = Wij.shape[0]
    Ns: int = Tmat.shape[0]
    Tij: np.ndarray = dt * beta * Wij

    Ps: np.ndarray = np.zeros((Ns, Npop))
    Pi: np.ndarray = np.zeros(Npop)
    Ps[0] = np.zeros(Npop) + 1 / Npop
    size: list = []
    t = 0
    while t < tmax:
        t += dt

        Phealthy = 1 - (Ps).sum(axis=0)
        Pi = (Ps.T * rbeta).sum(axis=1)

        prods: np.ndarray = np.array([np.prod((1 - tij * Pi)) for tij in Tij.T])
        trans = (1 - prods) * Phealthy

        Ps_new = Ps + np.dot(Tmat, Ps)
        Ps_new[0] += trans

        Ps = Ps_new

        size.append([list(Ps.sum(axis=1))])
    return size


class CompartmentalModel:
    def __init__(self, name, Wij, dt, rbeta, Tmat):
        self.name = name
        self.dt = dt
        self.rbeta = rbeta
        self.Tmat = Tmat * dt
        self.Wij = Wij
        self.Npop = Wij.shape[0]

    def run(self, beta, tmax, f=integrate_model):
        return f(self.Wij, self.Tmat, self.rbeta, self.dt, beta, tmax=tmax)

    def get_beta_for_size(self, size, tmax, beta0=1.0):
        def fsim(beta):
            s = self.run(beta, tmax)
            return (size - sum(s[-1][0])) ** 2

        # res = minimize(fsim, beta0, method='Nelder-Mead') #, tol=1e-6)
        res = sp.optimize.minimize(fsim, beta0, method="SLSQP", tol=1e-3)
        return res.x[0]

    def run_calsize(self, size, tmax, f=integrate_model):
        beta = float(self.size_interpolator(size))
        return self.run(beta, tmax, f)

    def calibrate_size(self, beta0, beta1, tmax=100):
        self.BETAS = np.exp(np.linspace(np.log(beta0), np.log(beta1), 20))
        self.SIZES = np.array(
            [sum(self.run(beta, tmax)[0][-1][0]) for beta in self.BETAS]
        )
        self.size_interpolator = sp.interpolate.interp1d(self.SIZES, self.BETAS)

    def calibrate_size_otherf(self, beta0, beta1, f, tmax=100):
        self.BETAS = np.exp(np.linspace(np.log(beta0), np.log(beta1), 20))
        self.SIZES = np.array(
            [sum(self.run(beta, tmax, f=f)[0][-1][0]) for beta in self.BETAS]
        )
        self.size_interpolator = sp.interpolate.interp1d(self.SIZES, self.BETAS)


class CompartmentalChainModel(CompartmentalModel):
    def __init__(self, name, Wij, dt, rbeta, rates):
        self.rates = rates  # *dt
        mRate = np.eye(rates.size) * self.rates
        Tmat = np.roll(mRate, 1).T - mRate
        super().__init__(name, Wij, dt, rbeta, Tmat)


def get_T(S, L):
    N = len(S)
    M = np.zeros((N, N))
    D = {s: i for i, s in enumerate(S)}
    for r, p, k in L:
        ir, ip = D[r], D[p]
        M[ir, ir] += -k
        M[ip, ir] += k
    return M


class CompartmentalModel_from_Reactions(CompartmentalModel):
    def __init__(self, name, Wij, dt, rbeta, states, list_of_reactions):
        Tmat = get_T(states, list_of_reactions)
        super().__init__(name, Wij, dt, rbeta, Tmat)
