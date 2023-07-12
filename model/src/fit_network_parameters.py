import itertools

from Network import Network
from SEIR import SEIR
# from NetworkSEIR3NPI import NetworkSEIR3NPI
# from src.model.PredefinedStrategy import NO_NPIS
# from run_multi_sim import timed_exp
import numpy as np


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

def run_by_network(network, params):
    parameters = [params[0]], C, EI_0, NO_MOBILITY, PREFIX  #NO_NPIS["npis"], 
    if network.name == Network.Regular.name:
        _, n_k = params
        return timed_exp(NetworkSEIR3NPI, Network.Regular, SEEDS, MODEL_SIZE, T_0_ALL, T, parameters, n=N, k=n_k)
    if network.name == Network.ER.name:
        _, p = params
        return timed_exp(NetworkSEIR3NPI, Network.ER, SEEDS, MODEL_SIZE, T_0_ALL, T, parameters, n=N, p=p)
    if network.name == Network.BA.name:
        _, m, p = params
        return timed_exp(NetworkSEIR3NPI, Network.BA, SEEDS, MODEL_SIZE, T_0_ALL, T, parameters, n=N, m=m, p=p)


def fit_parameters(network, ranges, y_true):
    errors = np.zeros(shape=len(ranges))
    y_preds = np.zeros(shape=(len(ranges), 1, T, 1))
    for idx in range(len(ranges)):
        _, y_pred, _, _ = run_by_network(network, ranges[idx])
        errors[idx] = compute_error(y_pred, y_true, T_0_ALL)
        y_preds[idx] = np.mean(y_pred, axis=0)
    print(errors)
    return np.argmin(errors), np.mean(y_true, axis=0), y_preds


def fit_regular(n_k_ranges):
    _, y_true, _, _ = standard_exp(BETA, T_0_STANDARD, T, N, EI_0, MODEL_SIZE, SEEDS)
    ranges = [(BETA/n_k, n_k) for n_k in n_k_ranges]
    return fit_parameters(Network.Regular, ranges, y_true)


def fit_er(phi, n_k, p_ranges):
    _, y_true, _, _ = run_by_network(Network.Regular, (phi, n_k))
    ranges = list(itertools.product([phi], p_ranges))
    return fit_parameters(Network.ER, ranges, y_true)


def fit_ba(phi, n_k, m_ranges, p_ranges):
    _, y_true, _, _ = run_by_network(Network.Regular, (phi, n_k))
    ranges = list(itertools.product(m_ranges, p_ranges))
    ranges = [(phi, p[0], p[1]) for p in ranges]
    return fit_parameters(Network.BA, ranges, y_true)

# Absolute value of Mean Bias Error (MBE)
def compute_error(y_pred, y_true, t_0):
    errors = []
    for seed in range(y_pred.shape[0]):
        error = 0
        for population in range(y_pred.shape[1]):
            # Sum error over all t
            error += np.sum(np.subtract(np.mean(y_pred[seed][population], axis=1), y_true)[t_0:]) + 1e-6
        errors.append(error)
    print("errors", errors)
    return abs(np.mean(errors))

if __name__ == "__main__":
    BETA = 2.7
    T = 100
    N = 17800
    EI_0 = [[3, 1]]
    MODEL_SIZE = 1
    SEEDS = np.arange(1, 4)
    C = 0.5
    NO_MOBILITY = np.array([[0.0]])
    PREFIX = [""]
    T_0_STANDARD = 7
    T_0_ALL = 0


    BETA = 0.78
    # 1 / days latent
    SIGMA = 1 
    # recovery rate = 1 / days infected
    GAMMA = 1/2


    n_k_ranges = np.arange(15, 26, 1)
    best_idx, y_true, y_preds = fit_regular(n_k_ranges)
    print(best_idx)
    print(y_preds)
    print(y_true)
    # param_labels = [["n_k: %.2f" % p] for p in n_k_ranges]
    # plot_fit_by_model(range(T), y_preds, y_true, param_labels, PREFIX, NO_NPIS, mark_idx=best_idx, t_0=0)

    # best_n_k = 22
    # phi = BETA/best_n_k

    # er_p_ranges = np.arange(0.0005, 0.0016, 0.0001)
    # best_idx, y_true, y_preds = fit_er(phi, best_n_k, er_p_ranges)
    # param_labels = [["er_p: %.4f" % p] for p in er_p_ranges]
    # plot_fit_by_model(range(T), y_preds, y_true, param_labels, PREFIX, NO_NPIS, mark_idx=best_idx, t_0=0)
    #
    best_er_p = 0.0012
    #
    # m_ranges = np.arange(3, 14, 1)
    # ba_p_ranges = [1]
    # best_idx, y_true, y_preds = fit_ba(phi, best_n_k, m_ranges, ba_p_ranges)
    # param_labels = [["m: %.2f, ba_p: %.2f" % p] for p in list(itertools.product(m_ranges, ba_p_ranges))]
    # plot_fit_by_model(range(T), y_preds, y_true, param_labels, PREFIX, NO_NPIS, mark_idx=best_idx, t_0=0)
    #
    best_m = 7
    best_ba_p = 1