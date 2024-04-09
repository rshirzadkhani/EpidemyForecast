import numpy as np
import networkx as nx
import statistics 
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

def statistical_characteristics(actives, cumulatives,names):
    T = actives[0].shape[1]
    all_durations = []

    temporal_max_actives = []
    temporal_final_attack = []
    temporal_peak_time = []
    temporal_duration  = []
    
    temp = len(actives)-1
    for seed in range(actives[temp].shape[0]):
        if np.std(actives[temp][seed,:]) != 0 :
        # Calculate temporal netrics:
            temporal_max_actives.append(max(actives[temp][seed,:]))
            temporal_final_attack.append(max(cumulatives[temp][seed, :]))
            peak_temp = np.where(actives[temp][seed,:]==max(actives[temp][seed, :]))
            temporal_peak_time.append(peak_temp[0][0])
            # idx_d = (np.where(new[temp][seed,:] < 0.005))[0]
            # peak_t = np.where(new[temp][seed,:]==max(new[temp][seed, :])) 
            # less, high = 0, 0
            # for i in idx_d:
            #     if i < peak_t[0][0]:
            #         less = i
            #     if i > peak_t[0][0]:
            #         high = i
            #         break
            # if high == 0:
            #     temporal_duration.append(T-less)
            # else:
            #     temporal_duration.append(high-less)
    print("Temporal max infection:", np.mean(temporal_max_actives), "+-", np.std(temporal_max_actives))
    print("Temporal peak time:", np.mean(temporal_peak_time), "+-", np.std(temporal_peak_time))
    print("Temporal final attack:", np.mean(temporal_final_attack), "+-", np.std(temporal_final_attack))
    # print("Temporal duration", np.mean(temporal_duration), "+-", np.std(temporal_duration))


    for ntw in range(0,len(actives)-1):
        print("Statistics for", names[ntw].name)

        max_actives = []
        final_attack = []
        peak_time = []
        duration  = []

        
        for seed in range(actives[ntw].shape[0]):
            if np.std(actives[temp][seed,:]) <= 0.00000001 or np.std(actives[ntw][seed, :]) <= 0.00000001:
                print(seed)
                a = 1
            else:
                print(max(actives[ntw][seed, :]))
                max_infect = abs(max(actives[ntw][seed, :]) - max(actives[temp][seed, :]))
                max_actives.append(max_infect)
                

                peak_temp = np.where(actives[temp][seed,:]==max(actives[temp][seed, :]))
                idx_p = np.where(actives[ntw][seed,:]==max(actives[ntw][seed, :]))        
                peak_time.append(abs(idx_p[0][0] - peak_temp[0][0]))#/peak_temp[0][0])
            # print(idx_p[0][0], peak_temp[0][0])
                if peak_temp[0][0] == 0:
                    print("i", seed)
                    print(actives[temp][seed,:])

            if np.std(actives[temp][seed,:]) >= 0.00000001 and np.std(actives[ntw][seed, :]) >= 0.00000001:
                f_a = abs(max(cumulatives[ntw][seed, :]) - max(cumulatives[temp][seed, :]))
                final_attack.append(f_a)


        #     idx_d = (np.where(new[ntw][seed,:] < 0.005))[0]
        #     peak_t = np.where(new[ntw][seed,:]==max(new[ntw][seed, :])) 
        #     less, high = 0, 0
        #     for i in idx_d:
        #         if i < peak_t[0][0]:
        #             less = i
        #         if i > peak_t[0][0]:
        #             high = i
        #             break
        #     if high == 0:
        #         duration.append(T-less)
        #     else:
        #         duration.append(high-less)
  
        # all_durations.append(duration)

        mean_actives = np.mean(max_actives)
        std_actives = np.std(max_actives)
        print("max infection", mean_actives, "+-", std_actives)
        # print(peak_time)
        mean_peak = np.mean(peak_time)
        std_peak = np.std(peak_time)
        print("peak time", mean_peak, "+-", std_peak)

        mean_attack = np.mean(final_attack)
        std_attack = np.std(final_attack)
        print("final attack", mean_attack, "+-", std_attack)

    # temps = temporal_duration #all_durations[3]
    # # print("Temporal duration", np.mean(temps), "+-", np.std(temps))
    # for i in range(len(actives)-1):
    #     durations = []
    #     static = all_durations[i]
    #     # print(temps)
    #     # print(static)
    #     for k in range(len(temps)):
    #         durations.append(abs(static[k]-temps[k])/temps[k])
    #     mean_duration = np.mean(durations)
    #     std_duration = np.std(durations)
    #     print("duration", mean_duration, "+-", std_duration)


def shortest_path(static, temporal):
    if temporal is not None:
        path = []
        for ntw in temporal:
            cc = sorted(nx.connected_components(ntw), key=len, reverse = True)
            # print("b", ntw)
            if len(cc) > 1:
                for i in range(1, len(cc)):
                    cc1 = cc[i]
                    for n in cc1:
                        ntw.remove_node(n)
            # print("a", ntw)

            path.append(nx.average_shortest_path_length(ntw))
        ave_path = statistics.mean(path)
    else:
        ave_path = nx.average_shortest_path_length(static)
    return ave_path

def modularity(static, temporal):
    if temporal is None:
        communities = nx.algorithms.community.louvain_communities(static, seed=123)
        print(len(communities))
        m=nx.algorithms.community.modularity(static, communities)
        print(m)
    else:
        m = []
        m1 = []
        for G in temporal:
            communities = nx.algorithms.community.louvain_communities(G, seed=123, resolution = 1)
            m1.append(len(communities))
            m.append(nx.algorithms.community.modularity(G, communities))
        print("mean", statistics.mean(m), statistics.stdev(m))
        print("mean1", statistics.mean(m1), statistics.stdev(m1))
    return 

def kl_div(ntw, names):
    temporal = ntw[4]
    for i in range(4):
        kl = []
        jsh = []
        for j in range(temporal.shape[0]):
            if np.std(ntw[i]) >= 0.000001 and np.std(ntw[i]) >= 0.000001:
                kl.append(entropy(ntw[i][j], temporal[j]))
        print(names[i].name, "mean:", statistics.mean(kl), "std dev:", statistics.stdev(kl))

def gephi_extract(static, temporal, ntw, path):
    if temporal is None:
        for i in [1,3,5]:
            nx.write_gefx(static[i], "./graph/gephi/"+path + ntw.name)

def global_efficiency(G):
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        lengths = nx.all_pairs_shortest_path_length(G, cutoff=5)
        g_eff = 0
        for source, targets in lengths:
            for target, distance in targets.items():
                if distance > 0:
                    g_eff += 1 / distance
        g_eff /= denom
    else:
        g_eff = 0
    return g_eff

def global_efficiency_calculater(G, temp_G):
    if temp_G is not None:
        d = []
        T = len(temp_G)
        for i in range(T):
            print(i)
            d.append(global_efficiency(temp_G[i]))
        return np.mean(d), "+-", np.std(d)
    else:
        return global_efficiency(G)
    

def algebraic_connectivity_calculator(G, temp_G):
    if temp_G is not None:
        d = []
        T = len(temp_G)
        for i in range(T):
            d.append(nx.algebraic_connectivity(temp_G[i], method='lanczos'))
        return np.mean(d), "+-", np.std(d)
    else:
        return nx.algebraic_connectivity(G)
    

def transitivity_calculator(G, temp_G):
    if temp_G is not None:
        d = []
        T = len(temp_G)
        for i in range(T):
            # print(i)
            d.append(nx.transitivity(temp_G[i]))
        return np.mean(d), "+-", np.std(d)
    else:
        return nx.transitivity(G)
    

def spectral_radius_calculator(G, temp_G):
    if temp_G is not None:
        d = []
        T = len(temp_G)
        for i in range(T):
            print(i)
            d.append(np.max(nx.adjacency_spectrum(temp_G[i], weight=None).real))
        return np.mean(d), "+-", np.std(d)
    else:
        return np.max(nx.adjacency_spectrum(G).real)