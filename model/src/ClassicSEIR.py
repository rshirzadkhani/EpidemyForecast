import numpy as np

from model.Constants import SEIR_COMPARTMENTS, SEIR_STATES


class ClassicSEIR:

    def __init__(self, n_0, seir_0, beta=0.6, sigma=0.5, symptomatic_rate=0.5, gamma=0.2, num_days=30):
        self.n_0 = n_0
        self.seir_0 = seir_0
        self.beta = beta
        self.sigma = sigma
        self.symptomatic_rate = symptomatic_rate
        self.gamma = gamma
        self.num_days = num_days
        self.seir = None
        self.population = None

    def run(self, mobility_matrix=None, stochastic=False):
        self.reset(mobility_matrix)
        for t in range(self.num_days - 1):
            new_exposed = self.get_new_exposed(t)
            new_infected_s, new_infected_a = self.get_new_infected(t, stochastic)
            new_recovered_s, new_recovered_a = self.get_new_recovered(t)
            self.update_seir(t, new_exposed, new_infected_s, new_infected_a, new_recovered_s, new_recovered_a, stochastic)
        seir = np.zeros((self.seir.shape[0], self.seir.shape[1], len(SEIR_COMPARTMENTS)))
        seir[:, :, 0] = self.seir[:, :, 0]
        seir[:, :, 1] = self.seir[:, :, 1]
        seir[:, :, 2] = self.seir[:, :, 2] + self.seir[:, :, 3]
        seir[:, :, 3] = self.seir[:, :, 4]
        return seir, self.population

    def update_seir(self, t, new_exposed, new_infected_s, new_infected_a, new_recovered_s, new_recovered_a, stochastic=False):
        self.seir[t + 1, :, 0] = self.seir[t, :, 0] - new_exposed
        self.seir[t + 1, :, 1] = self.seir[t, :, 1] + new_exposed - new_infected_s - new_infected_a
        self.seir[t + 1, :, 2] = self.seir[t, :, 2] + new_infected_s - new_recovered_s
        self.seir[t + 1, :, 3] = self.seir[t, :, 3] + new_infected_a - new_recovered_a
        self.seir[t + 1, :, 4] = self.seir[t, :, 4] + new_recovered_s + new_recovered_a
        self.population[t + 1] = self.population[t]

    def get_new_recovered(self, t):
        new_recovered_s = np.around(self.gamma_vec() * self.seir[t, :, 2])
        new_recovered_a = np.around(self.gamma_vec() * self.seir[t, :, 3])
        return new_recovered_s, new_recovered_a

    def get_new_exposed(self, t):
        susceptible = self.seir[t, :, 0]
        infected = self.seir[t, :, 2] + self.seir[t, :, 3]
        new_exposed = np.round(self.beta_vec() * susceptible * infected / self.population[t])
        return new_exposed

    def get_new_infected(self, t, stochastic=False):
        exposed = self.seir[t, :, 1]
        new_infected = np.round(self.sigma_vec() * exposed)
        new_infected_s = np.round(new_infected * self.symptomatic_rate)
        new_infected_a = new_infected - new_infected_s
        return new_infected_s, new_infected_a

    def gamma_vec(self):
        return np.full(shape=len(self.n_0), fill_value=self.gamma)

    def sigma_vec(self):
        return np.full(shape=len(self.n_0), fill_value=self.sigma)

    def beta_vec(self):
        return np.full(shape=len(self.n_0), fill_value=self.beta)

    def reset(self, mobility_matrix=None):
        self.seir = np.zeros((self.num_days, self.n_0.shape[0], len(SEIR_STATES)))
        self.seir[0] = self.seir_0.copy()
        self.population = np.zeros((self.num_days, self.n_0.shape[0]))
        self.population[0] = self.n_0.copy()
