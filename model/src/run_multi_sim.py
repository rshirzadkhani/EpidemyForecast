import random
import time
import uuid
import datetime
import numpy as np
import sys

from src.model.NetworkSEIR3NPI import NetworkSEIR3NPI
from src.model.MultiPopulationSEIR import MultiPopulationSEIR, SIGMA, GAMMA, N, K, ER_P, M, BA_P
from src.model.Network import Network
from src.model.PredefinedStrategy import CLOSING_HUBS, NO_NPIS, SOCIAL_DISTANCING, \
    WEARING_MASKS, TRAVEL_RESTRICTION, ARRIVAL_RESTRICTION, QUARANTINE_INFECTED, CONTACT_TRACING, \
    QUARANTINE_INFECTED_REOPEN, CONTACT_TRACING_REOPEN, SOCIAL_DISTANCING_REOPEN, CLOSING_HUBS_REOPEN, \
    WEARING_MASKS_REOPEN
from src.model.SEIR import SEIR
from src.model.StandardSEIR import StandardSEIR
from src.utils.PlotUtils import plot_daily_by_model, plot_cumulative_by_model
from src.utils.StatsUtils import print_peak, print_peak_timing, print_outbreak_size


START_DATE = datetime.datetime.strptime("2/22/20", "%m/%d/%y")
T0 = 0
T = 200
BETA = 2.7
PHI = BETA / K
C = 0.5
EI_0 = [[3, 1]]
# EI_0 = [[3, 1], [0, 0]]
MODEL_SIZE = len(EI_0)
MOBILITY_MATRIX = np.array([[0.0]])
# MOBILITY_MATRIX = np.array([[0, 0.1], [0.1, 0]])
PREFIX = [""]
# PREFIX = ["a", "b"]
# SEEDS = [3]
SEEDS = np.arange(0, 4)


def run_by_network(network, strategy):
    parameters = strategy["npis"], [PHI, PHI], C, EI_0, MOBILITY_MATRIX, PREFIX
    if network.name == Network.Regular.name:
        return timed_exp(NetworkSEIR3NPI, Network.Regular, SEEDS, MODEL_SIZE, T0, T, parameters, n=N, k=K)
    elif network.name == Network.ER.name:
        return timed_exp(NetworkSEIR3NPI, Network.ER, SEEDS, MODEL_SIZE, T0, T, parameters, n=N, p=ER_P)
    elif network.name == Network.BA.name:
        return timed_exp(NetworkSEIR3NPI, Network.BA, SEEDS, MODEL_SIZE, T0, T, parameters, n=N, m=M, p=BA_P)
    elif network.name == Network.WIFI1.name or network.name == Network.WIFI2.name or network.name == Network.WIFI3.name:
        return timed_exp(NetworkSEIR3NPI, network, SEEDS, MODEL_SIZE, T0, T, parameters)
    else:
        return standard_exp(BETA, 9, T, N, EI_0, MODEL_SIZE, SEEDS)


def run_exp_by_seed(model, network, seed, t0, t, parameters, **kwargs):
    npis, phis, c, ei_0, mobility_matrix, prefix = parameters
    random.seed(seed)
    np.random.seed(seed)
    model = MultiPopulationSEIR(model, network, t0, t, npis, phis, c, ei_0, mobility_matrix, prefix, **kwargs)
    seir, cum, r0, outbreak_size = model.run()
    return seir, cum, outbreak_size, r0


def timed_exp(model, network, seeds, model_size, t0, t1, parameters, **kwargs):
    seirs = np.zeros(shape=(len(seeds), model_size, t1, len(SEIR.__members__)))
    cum = np.zeros(shape=(len(seeds), model_size, t1, 1))
    outbreak_sizes = np.zeros(shape=(len(seeds), model_size))
    R0 = np.zeros(shape=(len(seeds), model_size))
    start = time.time()
    print(network.name)
    for idx, seed in enumerate(seeds):
        print("  - seed: ", seed)
        seirs[idx], cum[idx], outbreak_sizes[idx], R0[idx] = \
            run_exp_by_seed(model, network, seed, t0, t1, parameters, **kwargs)
    end = time.time()
    print("  - time elapsed: ", end - start)
    return seirs, cum, outbreak_sizes, R0


def standard_exp(beta, t0, t1, n, ei_0, model_size, seeds):
    seirs = np.zeros(shape=(1, model_size, t1, len(SEIR.__members__)))
    cum = np.zeros(shape=(1, model_size, t1, 1))
    outbreak_sizes = np.zeros(shape=(1, model_size))
    R0 = np.zeros(shape=(1, model_size))
    for i in range(model_size):
        model = StandardSEIR(beta=beta, sigma=SIGMA, gamma=GAMMA, t0=t0, t1=t1, n=n, ei_0=ei_0[i])
        seirs[0, i], cum[0, i], outbreak_sizes[0, i] = model.run()
        R0[0, i] = beta / GAMMA
    return np.repeat(seirs, len(seeds), axis=0), \
           np.repeat(cum, len(seeds), axis=0), \
           np.repeat(outbreak_sizes, len(seeds), axis=0), \
           np.repeat(R0, len(seeds), axis=0)


if __name__ == "__main__":
    transaction_id = str(uuid.uuid4())
    default_strategy = NO_NPIS
    networks = [Network.WIFI1]

    daily = np.zeros(shape=(len(networks), len(SEEDS), MODEL_SIZE, T, len(SEIR.__members__)))
    cumulative = np.zeros(shape=(len(networks), len(SEEDS), MODEL_SIZE, T, 1))
    outbreak_sizes = np.zeros(shape=(len(networks), len(SEEDS), MODEL_SIZE))
    for idx, network in enumerate(networks):
        daily[idx], cumulative[idx], outbreak_sizes[idx], _ = run_by_network(network, default_strategy)

    labels = [network.name for network in networks]
    print_peak(daily, default_strategy, labels, transaction_id)
    print_peak_timing(daily, default_strategy, labels, transaction_id)
    print_outbreak_size(outbreak_sizes, default_strategy, labels, transaction_id)
    plot_daily_by_model(range(T0, T), daily, SEIR.I, labels, default_strategy)
    plot_cumulative_by_model(range(T0, T), cumulative, labels, default_strategy)
