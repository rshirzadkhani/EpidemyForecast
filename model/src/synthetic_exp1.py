import pickle
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import datetime as dt

from ClassicSEIR import ClassicSEIR
from Constants import SEIR_COMPARTMENTS
from static_NetworkedSEIR import NetworkedSEIR
from plotting_utils import plot_synthetic_by_compartment, plot_degree_dist
from utils import compute_global_seir, compute_mean_confidence_interval


def construct_network(graph, n, sir_0):
    states = {}
    for i in graph.nodes:
        states[i] = {'state': 'S'}
    cc = sorted(nx.connected_components(graph), key=len, reverse=True)
    # print("num connected component:", len(cc))
    # print("size largest connected component:", len(cc[0]))
    node_indices = np.random.choice(list(cc[0]), np.sum(sir_0), replace=False)
    # node_indices = np.random.choice(range(n), np.sum(sir_0), replace=False)
    for i in np.arange(0, sir_0[0]):
        states[node_indices[i]] = {'state': 'E'}
    for i in np.arange(sir_0[0], sir_0[0] + sir_0[1]):
        states[node_indices[i]] = {'state': 'I_S'}
    for i in np.arange(sir_0[0] + sir_0[1], sir_0[0] + sir_0[1] + sir_0[2]):
        states[node_indices[i]] = {'state': 'I_A'}
    nx.set_node_attributes(graph, states)
    return graph


def synthetic_exp1():
    # mtl = pd.read_csv("dataset/stable/covid_SIR_montreal_infected_only.csv")
    # quebec = pd.read_csv("dataset/stable/contact_network/quebec_data.csv", skiprows=1, names=["date", "total", "active", "new"])
    # dates = pd.to_datetime(quebec.date).dt.strftime('%Y-%m-%d').tolist()
    
    dates = [dt.datetime.strftime(dt.datetime.strptime(d, '%Y-%m-%d'), "%m-%d") for d in pickle.load(open("model/mtl_dates.pkl", 'rb'))]
    start_date = dates[0]

    tr_sir = pickle.load(open("dataset/stable/contact_network/seir_mtl_100_updated.pkl", "rb"))
    wifi1_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2004-08-27_2006-11-30_GCC.edgelist", nodetype=int)
    wifi2_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2007-07-01_2008-02-26_GCC.edgelist", nodetype=int)
    wifi3_G = nx.read_edgelist("dataset/stable/contact_network/mtl_wifi_2009-12-02_2010-03-08_GCC.edgelist", nodetype=int)

    # timestamp
    T = tr_sir.shape[0]
    dates = pd.date_range(start = start_date, periods = T).strftime('%m-%d').tolist()

    base_e_0 = tr_sir[0, 2] * 2
    base_i_0 = tr_sir[0, 2]
    base_r_0 = tr_sir[0, 3]

    # transmission rate
    TRANSMISSIBILITY = 1
    # BETA = TRANSMISSIBILITY * 3
    BETA = 0.78
    # 1 / days latent
    SIGMA = 1 / 5
    # recovery rate = 1 / days infected
    GAMMA = 1 / 14
    # the probability of an exposed become symptomatic infected
    SYMPTOMATIC_RATE = 0.6
    # total population
    # n = 1.4E9
    # N = 14000
    n = 1780000
    N = 17800

    seeds = [3, 6, 13, 20, 21, 26, 43, 64, 97, 100]
    num_seeds = len(seeds)

    Regular_results = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    ER_results = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    BA_results = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results1 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results2 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))
    Real_results3 = np.zeros((num_seeds, T, len(SEIR_COMPARTMENTS)))

    K = 21 # for regular graph
    ER_P = 0.0014  # for ER graph
    M = 10  # for BA graph
    BA_P = 1 # for BA graph
    REAL_scale = 1
    tr_sir = tr_sir / n
    TRANSMISSIBILITY = BETA/K

    e_0 = int(np.round(base_e_0 / 100))
    i_0 = int(np.round(base_i_0 / 100))
    i_s_0 = int(np.round(i_0 * SYMPTOMATIC_RATE))
    i_a_0 = i_0 - i_s_0
    sir_0 = [e_0, i_s_0, i_a_0]

    for seed_idx, seed in enumerate(seeds):
        Regular_results[seed_idx] = regular_exp(T, N, K, BETA, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed) 

    for seed_idx, seed in enumerate(seeds):
        ER_results[seed_idx] = er_exp(T, N, ER_P, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed)

    base = base_exp(base_e_0, base_i_0, base_r_0, BETA, n, SIGMA, SYMPTOMATIC_RATE, GAMMA, T)

    for seed_idx, seed in enumerate(seeds):
        BA_results[seed_idx] = ba_exp(T, N, M, BA_P, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed)

    for seed_idx, seed in enumerate(seeds):
        Real_results1[seed_idx] = real_exp(T, N, wifi1_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed)

    for seed_idx, seed in enumerate(seeds):
        Real_results2[seed_idx] = real_exp(T, N, wifi2_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed)

    for seed_idx, seed in enumerate(seeds):
        Real_results3[seed_idx] = real_exp(T, N, wifi3_G, TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, seed)

    # pickle.dump([Regular_results, ER_results, base, BA_results, Real_results1, Real_results2, Real_results3], open("base_exp.pkl", "wb"))

    # [Regular_results, ER_results, base, BA_results, Real_results1, Real_results2, Real_results3] = pickle.load(open("base_exp.pkl", "rb"))
    # print(base.shape)
    plot_graphs(base, np.array([Regular_results, ER_results, BA_results, Real_results1, Real_results2, Real_results3]), tr_sir, dates[len(dates) - tr_sir.shape[0]:], t_match=15)

def plot_graphs(base, seir, real, dates, compartment="Infected", names=["Regular", "ER", "BA", "wifi 1", "wifi 2", "wifi 3", "Base", "Real"], t_match=0):
    mean_lst = []
    ci_lst = []
    seir = np.expand_dims(seir, axis=2)
    for idx in range(seir.shape[0]):
        mean, ci = compute_mean_confidence_interval(seir[idx])
        mean = np.squeeze(mean, axis=0)
        ci = np.squeeze(ci, axis=0)
        mean_lst.append(mean)
        ci_lst.append(ci)
    mean_lst = np.concatenate((np.array(mean_lst), base[None], real[None]), axis=0)
    ci_lst = np.concatenate((np.array(ci_lst), np.zeros(ci_lst[0].shape)[None], np.zeros(ci_lst[0].shape)[None]), axis=0)
    # print(mean_lst.shape, ci_lst.shape)
    # plot_synthetic_by_compartment(dates[:16], mean_lst[:,:16,:], compartment, names, ci_lst[:,:16,:], t_match)
    plot_synthetic_by_compartment(dates, mean_lst, compartment, names, ci_lst, t_match)

def real_exp(t, n, G, transmissibility, sigma, gamma, symptomatic_rate, sir_0, seed, scale=1):
    random.seed(seed)
    np.random.seed(seed)
    
    graph = G.copy()
    graph = construct_network(graph, n, sir_0)
    
    return network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, scale)

def base_exp(e_0, i_0, r_0, beta, n, sigma, symptomatic_rate, gamma, t):
    s_0 = n - e_0 - i_0 - r_0
    seir_0 = np.array([s_0, e_0, i_0 * symptomatic_rate, i_0 * (1 - symptomatic_rate), r_0])
    model2 = ClassicSEIR(np.array([n]), seir_0, beta=beta, sigma=sigma, symptomatic_rate=symptomatic_rate, gamma=gamma,
                         num_days=t)
    base_seir, base_pop = model2.run()
    base = compute_global_seir(base_seir, base_pop)
    return base


def regular_exp(t, n, k, beta, sigma, gamma, symptomatic_rate, sir_0, seed):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.random_regular_graph(k, n)
    print("regular number of edges", G.number_of_edges())
    graph = construct_network(G, n, sir_0)
    transmissibility = beta / k
    return network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)


def er_exp(t, n, p, transmissibility, sigma, gamma, symptomatic_rate, sir_0, seed):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.fast_gnp_random_graph(n, p)
    print("er number of edges", G.number_of_edges())
    # plot_degree_dist(G)
    graph = construct_network(G, n, sir_0)
    return network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)


def ba_exp(t, n, m, p, transmissibility, sigma, gamma, symptomatic_rate, sir_0, seed):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.dual_barabasi_albert_graph(n, m1=m, m2=1, p=p)
    print("ba number of edges", G.number_of_edges())
    graph = construct_network(G, n, sir_0)
    return network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph)


def network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, n, t, graph, scale=1):
    model1 = NetworkedSEIR(transmissibility=transmissibility, sigma=sigma, symptomatic_rate=symptomatic_rate,
                           gamma=gamma, num_days=t, scale=scale)
    seir = model1.run(graph)
    return seir


if __name__ == "__main__":
    start = time.time()
    synthetic_exp1()
    print("Time: ", time.time() - start)
