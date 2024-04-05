import time
import numpy as np
import os
from model.Constants import SEIR_COMPARTMENTS
# from model.ClassicSEIR import ClassicSEIR
from model.StandardSEIR import StandardSEIR
from model.plotting_utils import plot_for_contact_network, plot_edge_node
from model.utils import compute_mean_confidence_interval
from contact_network_utils import *
from NetworkUtils import *
from model.Network import Network, Data
from LoadNetwork import load_contact_network, load_data
from model.postprocessing import *
from model.DataPath import ResultPath
cur_dir = os.path.dirname(__file__)

def contact_network_exp(BETA, SIGMA, GAMMA, EI_0, data, network, seeds, save_path):
    SYMPTOMATIC_RATE = 0.6
    num_seeds = len(seeds)
    e_0 = EI_0[0]
    i_0 = EI_0[1]
    i_s_0 = int(np.round(i_0 * SYMPTOMATIC_RATE))
    i_a_0 = i_0 - i_s_0
    sir_0 = [e_0, i_s_0, i_a_0]

    '''
    This section plots average degree per snapshot for temporal networks
    '''
    # for dat in data:
    #     G, temp_G, T, dates, all_nodes = load_data(dat)
    #     degree = []
    #     for t1 in range(T):
    #         degree = network_density(G, temp_G, T, all_nodes)
    #     plot_edge_node(dates, degree, title=save_path+dat.name)


    G, temp_G, T, all_nodes = load_data(data, t_split=False)
    
    temporal_density = network_density(G, temp_G, T, all_nodes)
    TRANSMISSIBILITY = BETA / 22
    print("Temporal ave deg:", temporal_density, "phi:", TRANSMISSIBILITY)
    G, removing_nodes = delete_disconnected_components(G)
    all_nodes -= len(removing_nodes)
    print("Number of removed nodes:", len(removing_nodes))
    # seed_indices = select_initial_nodes(temp_G[0], sir_0, seeds)

    for ntw in network:
        print("Load "+ntw.name+" network...") 
        static_G, temporal_G = load_contact_network(ntw, G, temp_G, 0, 0)

    #     print(shortest_path(static_G, temporal_G))
        # print("Global Efficiency of", ntw, global_efficiency_calculater(static_G, temporal_G))
        # print("Algebraic Connectivity of", ntw, algebraic_connectivity_calculator(static_G, temporal_G))
        print("Transitivity of", ntw, transitivity_calculator(static_G, temporal_G))
        # print("Spectral Radius of", ntw, spectral_radius_calculator(static_G, temporal_G))


        # print("Run "+ntw.name+" experiments...") 
        # Network_results = np.zeros((num_seeds, T+1, len(SEIR_COMPARTMENTS)))
        # New_Network_results = np.zeros((num_seeds, T+1, len(SEIR_COMPARTMENTS)-1))
        # Cum_Network_results = np.zeros((num_seeds, T+1, 2))
        # if ntw.name in [Network.STANDARD_GRAPH.name, Network.ER.name]:
        #     seed_indices = select_initial_nodes(static_G, sir_0, seeds)
        # for seed_idx, seed in enumerate(seeds):
        #     print("seed: ", seed)
        #     if ntw.name in [Network.ER.name, Network.BA.name]:
        #         static_G, temporal_G = load_contact_network(ntw, G, temp_G, seed, 0)

        #     Network_results[seed_idx] , New_Network_results[seed_idx], Cum_Network_results[seed_idx] = \
        #         run_with_network(T, static_G, temporal_G,
        #                          TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0, 
        #                          seed_indices[seed_idx], all_nodes, removing_nodes,
        #                          find_t_20_percent=False, prediction_mode=False)

        # np.save("./output/"+save_path+ntw.name+"_active_cases", Network_results)
        # np.save("./output/"+save_path+ntw.name+"_new_cases", New_Network_results)
        # np.save("./output/"+save_path+ntw.name+"_cum_cases", Cum_Network_results)


def contact_network_exp_predict(BETA, SIGMA, GAMMA, EI_0, data, network, seeds, save_path, predict):
    SYMPTOMATIC_RATE = 0.6
    num_seeds = len(seeds)
    e_0 = EI_0[0]
    i_0 = EI_0[1]
    i_s_0 = int(np.round(i_0 * SYMPTOMATIC_RATE))
    i_a_0 = i_0 - i_s_0
    sir_0 = [e_0, i_s_0, i_a_0]
    G, temp_G, T, all_nodes = load_data(data)
    TRANSMISSIBILITY = BETA / 22
    # G, removing_nodes = delete_disconnected_components(G)
    # all_nodes -= len(removing_nodes)
    # print("removing nodes step 1:",len(removing_nodes))
    # seed_indices = select_initial_nodes(temp_G[0], sir_0, seeds)
    
    # t_split = run_with_network(T, G, temp_G, TRANSMISSIBILITY, SIGMA,
    #                              GAMMA, SYMPTOMATIC_RATE, sir_0, seed_indices[0], 
    #                              all_nodes, removing_nodes, prediction_mode=True)
    
    t_split = int(T * 0.5)
    print("Number of Connected Components: {}".format(nx.number_connected_components(G)))
    G, temp_G, T, all_nodes = load_data(data, t_split=t_split)
    G, removing_nodes = delete_disconnected_components(G)
    all_nodes -= len(removing_nodes)
    print("removing nodes step 2:",len(removing_nodes))
    seed_indices = select_initial_nodes(temp_G[0], sir_0, seeds)
    for ntw in network:
        print("Load "+ntw.name+" network...") 
        static_G, temporal_G = load_contact_network(ntw, G, temp_G, 0, t_split-1)
        if ntw.name == Network.STATIC.name:
            static_G = creat_exponential_threshold_graph(temp_G[0: t_split], 0.26, 1)
        
        
        if temporal_G is not None:
            temporal_G = temp_G[t_split: T+1]
        # print("Global Efficiency of", ntw, global_efficiency_calculater(static_G, temporal_G))
        # print("Algebraic Connectivity of", ntw, algebraic_connectivity_calculator(static_G, temporal_G))      
        print("Run "+ntw.name+" experiments...") 
        experiment_length = T - t_split
        
        Network_results = np.zeros((num_seeds, experiment_length + 1 , len(SEIR_COMPARTMENTS)))
        New_Network_results = np.zeros((num_seeds, experiment_length + 1, len(SEIR_COMPARTMENTS)-1))
        Cum_Network_results = np.zeros((num_seeds, experiment_length + 1, 2))

        for seed_idx, seed in enumerate(seeds):
            print("seed: ", seed)
            Network_results[seed_idx] , New_Network_results[seed_idx], Cum_Network_results[seed_idx] = \
                run_with_network(experiment_length, static_G, temporal_G, 
                                 TRANSMISSIBILITY, SIGMA, GAMMA, SYMPTOMATIC_RATE, sir_0,
                                 seed_indices[seed_idx], all_nodes, removing_nodes, t_split=t_split, 
                                 find_t_20_percent=False, prediction_mode=predict)

        np.save("./output/"+save_path+ntw.name+"_active_cases", Network_results)
        np.save("./output/"+save_path+ntw.name+"_new_cases", New_Network_results)
        np.save("./output/"+save_path+ntw.name+"_cum_cases", Cum_Network_results)


def plot_graphs(seir, dates, filename, networks=None):
    mean_lst = []
    lower_lst = []
    upper_lst = []

    # seir = np.expand_dims(seir, axis=2)
    # seir = np.expand_dims(seir, axis=3)
    for idx in seir:
        # print(idx)
        # mean, ci = compute_mean_confidence_interval(seir[idx])
        mean, lower, upper = compute_mean_confidence_interval(idx)
        # mean = np.squeeze(mean, axis=0)
        # ci = np.squeeze(ci, axis=0)
        mean_lst.append(mean)
        lower_lst.append(lower)
        upper_lst.append(upper)
    # mean_lst=np.squeeze(mean_lst)
    # ci_lst = np.squeeze(ci_lst)

    plot_for_contact_network(dates, mean_lst, lower_lst, upper_lst, filename, labels=networks)


def plot_experiments(data, network, path):
    print("Plot experiments for "+data.name+"...")
    d = {}
    with open("./output/"+path+"/stats.txt") as f:
        for line in f:
            (key, val) = line.split(" ")
            d[key] = int(val)
    density = []
    active_infected = []
    new_infected = []
    cum_infected = []
    
    for ntw in network:
        active_infected_i = []
        new_infected_i = []
        cum_infected_i = []
        actives = np.load("./output/{}/forecasting/{}_{}_active_cases.npy".format(path, path, ntw.name))
        # new = np.load("./output/" + path + ntw.name + "_new_cases.npy")
        cum = np.load("./output/{}/forecasting/{}_{}_cum_cases.npy".format(path, path, ntw.name))
        for i in range(actives.shape[0]):
            if np.std(actives[i,:,2]) >= 0.000001:
                active_infected_i.append(actives[i,:,2])
                # new_infected.append(new[:,:,1])
                cum_infected_i.append(cum[i,:,0])
        # print(actives[:,:,2].shape)
        active_infected.append(np.array(active_infected_i))
        cum_infected.append(np.array(cum_infected_i))
        new_infected.append(np.array(new_infected_i))
    print(active_infected[0].shape)
    print(active_infected[1].shape)
    print(active_infected[2].shape)
    print(active_infected[3].shape)
    # statistical_characteristics(active_infected,cum_infected, network)
    # kl_div(active_infected, network)

    dates = list(range(d["T"])) 
    graph_label = [("Full static"),("DegMST"),("EdgeMST"),("Temporal")]
    # graph_label = [("Standard"),("Regular"), ("Random"),("Temporal")]

    
    plot_graphs(active_infected, dates, "./graph/"+path+"/forecasting/active_cases_"+str(data_i.name), networks=graph_label) 
    plot_graphs(cum_infected, dates, "./graph/"+path+"/forecasting/cum_cases_"+str(data_i.name), networks=graph_label) 
    


if __name__ == "__main__":
    BETA = 2.7
    SIGMA = 1/5
    GAMMA = 1/14
    # EI_0 = [3, 1]
    # EI_0 = [30, 10]
    EI_0 = [1200, 400]
    SEED = range(50)
    # standard_SEIR(BETA, SIGMA , GAMMA, 0, 29, 692, [EI_0], SEED, save_path = "school/figure_2/school_")

    # [Data.SCHOOL, Data.COP1, Data.COP2, Data.COP3, Data.SFHH, Data.WIFI, Data.SAFEGRAPH]
    # [Data.HIGHSCHOOL, Data.LYONSCHOOL, Data.CONFERENCE, Data.WORKPLACE]
    datasets = [Data.SAFEGRAPH]

    exp_ntw_list = [Network.STATIC, Network.MST_D_MATCH, 
                  Network.MST_W_MATCH, Network.TEMPORAL]
    ntw_list_1 = [Network.STATIC, Network.MST_D_MATCH, 
                  Network.MST_W_MATCH, Network.TEMPORAL]
    ntw_list_2 = [Network.STATIC, Network.STANDARD_GRAPH, Network.ER, 
                  Network.TEMPORAL]
    
    ###################### Run experiments fig 3 ######################

    for i, data_i in enumerate(datasets):
        path = ResultPath[str(data_i.name)]
        full_path = path+"/forecasting/"+path+"_"
        # full_path = path+"/predict_cases_maxst/"+path+"_"
        # full_path = path+"/new_1227/"+path+"_"
        # contact_network_exp_predict(BETA, SIGMA, GAMMA, EI_0, data = data_i, seeds=SEED, 
        #                     network = exp_ntw_list, save_path = full_path, predict=True)

    ###################### Plot experiments fig 3 ######################
    
    for i, data_i in enumerate(datasets):
        path = ResultPath[str(data_i.name)]
    #     read_paths = [path+"/predict_cases_minst/"+path+"_", 
    #                   path+"/new_1227/"+path+"_"]
    #     save_path = path+"/final_figures/"+path+"_"
    #     fnames = ["fig4", "fig3"]
    #     for i in range(len(read_paths)):
        plot_experiments(data = data_i , network = ntw_list_1, 
                            path = path)
                             
    
    ###################### Plot experiments fig 2 ######################
    
    # for i, data_i in enumerate(datasets):
    #     path = ResultPath[str(data_i.name)]
    #     plot_experiments(data = data_i , network = ntw_list_2, path = paths[i], fname = "forecasting")                                                            
