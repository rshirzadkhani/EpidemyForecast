import numpy as np
from model.SEIR import SEIR


class StandardSEIR:

    def __init__(self, beta, sigma, gamma, t0, t1, **kwargs):
        self.t0 = t0
        self.t1 = t1
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.n = kwargs["n"]
        self.ei_0 = kwargs["ei_0"]
        self.final_outbreak_size = 0

    def run(self):
        print(self.t0, self.t1)
        seir_0 = np.zeros(len(SEIR.__members__))
        seir_0[0] = self.n - self.ei_0[0] - self.ei_0[1]
        seir_0[1] = self.ei_0[0]
        seir_0[2] = self.ei_0[1]
        seirs = np.zeros(shape=(self.t1, len(SEIR.__members__)))
        seirs[self.t0 - 1] = seir_0
        cum = np.zeros(shape=(self.t1, 1))
        for t in np.arange(self.t0, self.t1):
            seirs[t], daily = self.update(seirs[t - 1])
            cum[t] = daily + cum[t - 1]
        return seirs, cum, self.final_outbreak_size

    def update(self, x):
        s, e, i, r = x
        daily = self.sigma * e
        ds = - self.beta * s * i / self.n
        de = self.beta * s * i / self.n - daily
        di = daily - self.gamma * i
        dr = self.gamma * i
        self.final_outbreak_size += self.sigma * e
        # print(de, di, dr)
        return [s + ds, e + de, i + di, r + dr], daily
