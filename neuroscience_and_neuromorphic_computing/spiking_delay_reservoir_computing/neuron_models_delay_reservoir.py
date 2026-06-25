import numpy as np
import scipy as scy
from numba import njit
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os





class LIF():
    def __init__(self, V_rest, RC, V_th, K_fb,K_inj, J):
        self.V_rest = V_rest
        self.RC = RC
        self.V_th = V_th
        self.K_fb = K_fb
        self.K_inj = K_inj
        self.J = J

    def dx_dt(self, K_X, X, X_tau):
        #NOTE: using self.RC, not RC
        K_X[0] = ((X[0] - self.V_rest) / self.RC) + self.K_fb * X_tau  + self.K_inj*self.J
        return K_X

    def threshold(self, X):
        if X[0] >= self.V_th:
            X[0] = self.V_rest
        return X


class FHN():
    def __init__(self, alpha, beta, inv_gamma, V_th, V_reset, K_fb, K_inj, J):
        self.alpha = alpha
        self.beta = beta
        self.inv_gamma = inv_gamma
        self.V_th = V_th
        self.V_reset = V_reset
        self.K_fb = K_fb
        self.K_inj = K_inj
        self.J = J

    def dx_dt(self, K_X, X, X_tau):
        V, W = X
        V_tau = X_tau
        K_X[0] = V - V**3 / 3 - W + self.K_fb* V_tau + self.K_inj*self.J
        K_X[1] = self.inv_gamma * (V + self.alpha - self.beta * W)
        return K_X

    #def threshold(self, X):
    #    if X[0] >= self.V_th:
    #        X[0] = self.V_reset
    #    return X


class Izhikevich():
    def __init__(self, a, b, c, d, V_th, K_fb, K_inj, J):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V_th = V_th
        self.K_fb = K_fb
        self.K_inj = K_inj
        self.J = J

    def dx_dt(self, K_X, X, X_tau):
        v, u = X
        v_tau = X_tau
        K_X[0] = 0.04 * v**2 + 5 * v + 140 - u + self.K_fb* (v_tau - v) + self.K_inj*self.J
        K_X[1] = self.a * (self.b * v - u)
        return K_X

    def threshold(self, X):
        if X[0] >= self.V_th:
            X[0] = self.c
            X[1] = X[1] + self.d
        return X


class HH():
    def __init__(self, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, V_th, V_reset, K_fb, K_inj, J):
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.V_th = V_th
        self.V_reset = V_reset
        self.K_fb = K_fb
        self.K_inj = K_inj
        self.J = J

    def alpha_m(self, V):
        x = V + 40.0
        if abs(x) < 1e-8:
            return 1.0
        return 0.1 * x / (1.0 - np.exp(-x / 10.0))

    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        x = V + 55.0
        if abs(x) < 1e-8:
            return 0.1
        return 0.01 * x / (1.0 - np.exp(-x / 10.0))

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def dx_dt(self, K_X, X, X_tau):
        # X = [V, m, h, n]
        V, m, h, n = X
        V_tau = X_tau

        I_Na = self.g_Na * m**3 * h * (V - self.E_Na)
        I_K = self.g_K * n**4 * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)

        K_X[0] = (self.K_fb*V_tau + self.K_inj*self.J - I_Na - I_K - I_L) / self.C_m
        K_X[1] = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        K_X[2] = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        K_X[3] = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n

        return K_X



def get_neuron_model(model):
    """
    model = 1 -> LIF
    model = 2 -> FHN
    model = 3 -> Izhikevich
    model = 4 -> Hodgkin-Huxley
    """

    if model == 1:
        params = {
            "V_rest": 0.0,
            "RC": 0.2,
            "V_th": 1.0,
            "K_fb": 0.4,
            "K_inj": 2.0,
            "J":0.0
        }

        X0 = np.array([0.0], dtype=float)
        neuron = LIF(**params)
        name = "Leaky Integrate-and-Fire"

    elif model == 2:
        params = {
            "alpha": 0.7,
            "beta": 0.8,
            "inv_gamma": 1 / 12.5,
            "V_th": 1.0,
            "V_reset": 0.0,
            "K_fb": 0.4,
            "K_inj": 2.0,
            "J":0.0
        }

        X0 = np.array([0.0, 0.0], dtype=float)
        neuron = FHN(**params)
        name = "FitzHugh-Nagumo"

    elif model == 3:
        params = {
            "a": 0.02,
            "b": 0.2,
            "c": -50.0,
            "d": 2.0,
            "V_th": 30.0,#Threshold imposed in original model
            "K_fb": .5,
            "K_inj": 10.0,
            "J":0.0
        }

        X0 = np.array([-70.0, -14.0], dtype=float)
        neuron = Izhikevich(**params)
        name = "Izhikevich"

    elif model == 4:
        params = {
            "C_m": 1.0,
            "g_Na": 120.0,
            "g_K": 36.0,
            "g_L": 0.3,
            "E_Na": 50.0,
            "E_K": -77.0,
            "E_L": -54.0,
            "V_th": 30.0,
            "V_reset": -65.0,
            "K_fb": 1.0,
            "K_inj": 1.0,
            "J":0.0
        }

        # X = [V, m, h, n]
        X0 = np.array([-65.0, 0.0529, 0.5960, 0.3177], dtype=float)
        neuron = HH(**params)
        name = "Hodgkin-Huxley"

    else:
        raise ValueError("model must be 1, 2, 3, or 4")

    return neuron, X0, params, name



def RK4(neuron, dt, dt_half, X, X_tau, X_next, K_X_1, K_X_2, K_X_3, K_X_4):

    K_X_1[:] = neuron.dx_dt(K_X_1, X, X_tau)
    K_X_2[:] = neuron.dx_dt(K_X_2, X + dt_half * K_X_1, X_tau)
    K_X_3[:] = neuron.dx_dt(K_X_3, X + dt_half * K_X_2, X_tau)
    K_X_4[:] = neuron.dx_dt(K_X_4, X + dt * K_X_3, X_tau)

    X_next = X + (dt / 6.0) * (K_X_1 + 2*K_X_2 + 2*K_X_3 + K_X_4)

    return X_next


def Euler(neuron, dt_half, dt, X, X_tau, X_next, K_X_1, K_X_2, K_X_3, K_X_4):

    K_X_1[:] = neuron.dx_dt(K_X_1, X, X_tau)

    X_next = X + dt *K_X_1

    return X_next

