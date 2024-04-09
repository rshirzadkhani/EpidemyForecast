import numpy as np
import networkx as nx
import random
import statistics 
from model.Constants import SEIR_COMPARTMENTS
from NetworkedSEIR import NetworkedSEIR
from NetworkedSEIR_predict import NetworkedSEIR_p
from model.StandardSEIR import StandardSEIR
from model.SEIR import SEIR

def select_initial_nodes(graph, sir_0, seeds):
    
    node_indices = []
    cc = sorted(nx.connected_components(graph), key=len, reverse=True)
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        node_indices.append(np.random.choice(list(cc[0]), np.sum(sir_0), replace=False))
    # print(node_indices)
    return node_indices

def construct_network(static_graph, temporal_graph, sir_0, node_indices):
    states = {}
    for i in static_graph.nodes:
        states[i] = {'state': 'S'}
    # if temporal_graph is not None:
    #     # cc1 = max(nx.connected_components(temporal_graph), key=len)
    #     cc = sorted(nx.connected_components(temporal_graph), key=len, reverse=True)
    #     # print("cc",cc,"cc1",cc1)
    # else:
    #     # cc1 = max(nx.connected_components(static_graph), key=len)
    #     cc = sorted(nx.connected_components(static_graph), key=len, reverse=True)
    # node_indices = np.random.choice(list(cc[0]), np.sum(sir_0), replace=False)
    # node_indices = np.random.choice(range(n), np.sum(sir_0), replace=False)
    for i in np.arange(0, sir_0[0]):
        states[node_indices[i]] = {'state': 'E'}
    for i in np.arange(sir_0[0], sir_0[0] + sir_0[1]):
        states[node_indices[i]] = {'state': 'I_S'}
    for i in np.arange(sir_0[0] + sir_0[1], sir_0[0] + sir_0[1] + sir_0[2]):
        states[node_indices[i]] = {'state': 'I_A'}
    nx.set_node_attributes(static_graph, states)
    # print(node_indices)
    return static_graph

def network_density( static_G, temporal_G, t, all_nodes):
    degree = []
    if temporal_G is None:
        num_edges = static_G.number_of_edges()
        degree.append(num_edges *2 / all_nodes)
    else:
        for t1 in range(t):
            num_edges = temporal_G[t1].number_of_edges()
            degree.append(num_edges*2/ all_nodes)
    
    # density = lambda m,n : 2*m/(n*(n-1))
    # d = []
    # if temporal_G is None:
    #     m1 = static_G.number_of_edges()
    #     # n1 = static_G.number_of_nodes()
    #     d.append(density (m1,all_nodes))
    # else: 
    #     for t1 in range(t):
    #         m1 = temporal_G[t1].number_of_edges()
    #         # n1 = temporal_G[t1].number_of_nodes()
    #         d.append(density(m1,all_nodes))
    # print(d)
    return round(statistics.mean(degree))

def run_with_network(t, G, temp_G, transmissibility, sigma, gamma, symptomatic_rate, 
                     sir_0, seed, all_nodes, removing_nodes, scale=1, npi=None, 
                     t_apply_npi=None, t_open_up=None, npi_reopen =None, t_split=0,
                     find_t_20_percent=True, prediction_mode=True, **kwargs):
    # random.seed(seed)
    # np.random.seed(seed)
    # print(seed)
    initial_graph = None
    if temp_G is not None:
        initial_graph = temp_G[0]
    graph = G.copy()
    graph = construct_network(graph, initial_graph, sir_0, seed)
    
    seir = np.zeros(( t, len(SEIR_COMPARTMENTS)))

    # if npi is None:
    #     seir, new_cases, cum_cases = network_exp_with_seed(transmissibility, sigma, symptomatic_rate, 
    #                                                        gamma, t, graph, temp_G,all_nodes,removing_nodes, 
    #                                                        scale, find_t_20_percent, prediction_mode)
    # elif t_open_up is None:
    #     seir, new_cases = network_exp_with_npi(transmissibility, sigma, symptomatic_rate, gamma, t, graph, temp_G, npi, t_apply_npi,removing_nodes, scale, **kwargs)
    # else:
    #     seir = network_exp_with_reopen(transmissibility, sigma, symptomatic_rate, gamma, t, graph, npi, t_apply_npi, t_open_up, npi_reopen, scale, **kwargs)
    # return seir, new_cases, cum_cases
    return network_exp_with_seed(transmissibility, sigma, symptomatic_rate, 
                                                           gamma, t, graph, temp_G,all_nodes,removing_nodes, 
                                                           t_split, find_t_20_percent, prediction_mode)


    
def network_exp_with_seed(transmissibility, sigma, symptomatic_rate, gamma, t, graph, temp_G,all_nodes,
                          removing_nodes, t_split, find_t_20_percent, prediction_mode):
    if prediction_mode:
        # print("Prediction mode")
        model = NetworkedSEIR_p(transmissibility=transmissibility, sigma=sigma, symptomatic_rate=symptomatic_rate,
                            gamma=gamma, duration=t, all_nodes=all_nodes, t_split=t_split,
                            removing_nodes=removing_nodes, find_t_20_percent=find_t_20_percent)
        # if find_t_20_percent:
        #     t_split = model.run(graph, temp_G)
        # else:
        #     seir, new_cases, cum_cases = model.run(graph, temp_G)
    else:
        # print("Base mode")
        model = NetworkedSEIR(transmissibility=transmissibility, sigma=sigma, symptomatic_rate=symptomatic_rate,
                            gamma=gamma, duration=t, all_nodes=all_nodes,
                            removing_nodes=removing_nodes)

        # seir, new_cases, cum_cases = model.run(graph, temp_G)
    return model.run(graph, temp_G) #seir, new_cases, cum_cases

def network_exp_with_npi(transmissibility, sigma, symptomatic_rate, gamma, t, graph, temp_G, npis, t_apply_npi,removing_nodes, scale=1, no_npi_nodes=None, **kwargs):#pickle.load(open("no_npi_nodes.pkl", "rb")), **kwargs):
    model1 = NetworkedSEIR(transmissibility=transmissibility, sigma=sigma, symptomatic_rate=symptomatic_rate,
                           gamma=gamma, duration=t, scale=scale, no_npi_nodes=no_npi_nodes,removing_nodes=removing_nodes)
    seir, new_cases = model1.run_npi(graph, temp_G, npis, t_apply_npi, **kwargs)
    return seir, new_cases


def standard_SEIR(beta, sigma, gamma, t0, t1, n, ei_0, seed, save_path):
    seirs = np.zeros(shape=(len(seed), t1, len(SEIR_COMPARTMENTS)))
    new = np.zeros(shape=(len(seed), t1+1, len(SEIR_COMPARTMENTS)-1))
    cum = np.zeros(shape=(len(seed), t1, 1))  
    for i in range(len(seed)):  
        model = StandardSEIR(beta, sigma, gamma, t0=t0, t1=t1, n=n, ei_0=ei_0[0])
        seirs[i], cum[i], _ = model.run()
    np.save("./output/"+save_path+"STANDARD_EQ_active_cases", seirs / n)
    # np.save("./output/"+save_path+"STANDARD_EQ_new_cases", new / n)
    # np.save("./output/"+save_path+"STANDARD_EQ_cum_cases", cum / n)


def delete_disconnected_components(graph):
    cc = sorted(nx.connected_components(graph), key=len, reverse=True)
    graph1 = graph.subgraph(max(nx.connected_components(graph), key=len))
    removing_nodes = []
    if len(cc) > 1:
        for i in range(1, len(cc)):
            a1 = list(cc[i])
            for n in a1:
                removing_nodes.append(n)
    return graph1, removing_nodes

def edges_and_nodes_per_time(graph):
    edges = []
    nodes = []
    for temporal_net in graph:
        edges.append(temporal_net.number_of_edges())
        nodes.append(temporal_net.number_of_nodes())
    return edges, nodes

def matching_degree_top_edge(main_G, init_G, degree):
    # print(top_G.number_of_edges() *2 / main_G.number_of_nodes())
    # main_G = nx.from_pandas_edgelist(edgelist,'user_a',"user_b", ['weight'])
    edges = sorted(main_G.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True)
    num_edges = main_G.number_of_edges()
    curr_degree = num_edges *2 / main_G.number_of_nodes()
    remaining_edges = len(edges)/5
    # print(curr_degree, degree, remaining_edges)
    while round(curr_degree) != degree :
        new_edges = [edges[i] for i in range(int(remaining_edges))]
        top_G = init_G.copy()
        top_G.add_edges_from(new_edges)
        curr_degree = top_G.number_of_edges() *2 / main_G.number_of_nodes()
        # print("curr", curr_degree)
        if round(curr_degree) > degree:
            remaining_edges = len(new_edges) * 1 / 3
        else:
            remaining_edges = len(new_edges) * 3 / 2
    # print(top_G, curr_degree)
    cc=sorted(nx.connected_components(top_G), key=len, reverse=True)
    return top_G

def node_degree_count(temp_G, init_G):
    node_dict = {}
    for node in init_G.nodes():
        node_dict[node] = 0
    for snapshot in temp_G:
        degree_list = snapshot.degree()
        for node, deg in degree_list:
            if node not in node_dict.keys():
                continue
            node_dict[node] += deg / len(temp_G)
    node_dict = dict(sorted(node_dict.items(), key=lambda item: item[1], reverse=True))
    node_list = list(node_dict.keys())
    return node_list

def matching_degree_top_node(node_list, main_G, init_G, degree):
    # degree_sequence = pd.DataFrame(main_G.degree, columns=["node_id", "degree"])
    # degree_sequence = degree_sequence.sort_values(by=["degree"], ascending=False)
    edges = main_G.edges()
    curr_degree = 0
    c = 0
    top_G = init_G
    while round(curr_degree) != degree :
        # selected_node = degree_sequence["node_id"].iloc[c]
        selected_node = node_list[c]
        # new_edges = main_G.edges(selected_node)
        new_edges = [x for x in edges if x[0] == selected_node or x[1] == selected_node]
        top_G.add_edges_from(new_edges)
        curr_degree = top_G.number_of_edges() *2 / main_G.number_of_nodes()
        if round(curr_degree) > degree:
            return prev_G
        else:
            c += 1
        prev_G = top_G.copy()
    return top_G

def random_selection(G):
    nodes = list(G.nodes)
    # edges_list = list(G.edges)
    selected_nodes = []
    edges = []
    for i in nodes:
        # if i not in selected_nodes:
        # print(i)
        edges_list = list(G.edges(i))
        random.seed(123)
        np.random.seed(123)
        a = random.randrange(0,len(edges_list))
        edges.append(edges_list[a])
        selected_nodes.append((edges_list[a])[0])
        selected_nodes.append((edges_list[a])[1])

    new_graph = nx.Graph()
    new_graph.add_edges_from(edges)
    print(new_graph)
    cc=sorted(nx.connected_components(new_graph), key=len, reverse=True)
    print(len(cc))
    return new_graph

def MST(G):
    T =nx.minimum_spanning_tree(G, weight='weight')
    # print(T)
    # T = nx.maximum_spanning_tree(G, weight='weight')
    # print(T1)
    return T


def maximum_node_degree(graph, temporal=True):
    max_degree_list = []
    if temporal:
        for snapshot in graph:
            max_degree = 0
            # degree_list.append(max(sorted(snapshot.degree(), reverse=True)))
            degree_list = snapshot.degree()
            for _, deg in degree_list:
                if deg > max_degree:
                    max_degree = deg
            max_degree_list.append(max_degree)
    
    else:
        max_degree = 0
        degree_list = graph.degree()
        for _, deg in degree_list:
            if deg > max_degree:
                max_degree = deg
        max_degree_list.append(max_degree)

    return np.mean(max_degree_list), np.std(max_degree_list)


def number_of_connected_components(graph, temporal=True):
    if temporal:
        temp_cc =[]
        for snapshot in graph:
            # temp_cc.append(nx.number_connected_components(snapshot))
            cc = sorted(nx.connected_components(snapshot), key=len, reverse=True)
            print(list(cc))
        # print(temp_cc)
        # ncc = np.mean(temp_cc)

    else:
        ncc = nx.number_connected_components(graph)
    return ncc
        

def creat_exponential_threshold_graph(temp_G, omega, tau):
    # T = len(temp_G)
    
    edges = {}
    G = nx.Graph()
    for t, snapshot in enumerate(temp_G):
        print("processing snapshot: ", t)
        # print(snapshot)
        edgelist = snapshot.edges()
        # print(edgelist)
        for (u,v) in edgelist:
            if (u,v) in edges or (v,u) in edges:
                edges[min(u,v), max(u, v)] += np.exp(-t/tau)
                # G[min(u, v)][max(u, v)]["weight"] += np.exp(-t/tau)
            else:
                edges[min(u,v), max(u, v)] = np.exp(-t/tau)
                # G.add_edge(min(u, v), max(u, v), weight = np.exp(-t/tau))
    print("adding edges to static graph")
    for (u, v), weight in edges.items():
        if weight >= omega:
            G.add_edge(u, v)
    print("processed graph: " , G)
    return G



if __name__ == "__main__":
    from model.Network import Network, Data
    from LoadNetwork import load_contact_network, load_data

    datasets = [Data.SCHOOL, Data.WORKPLACE, Data.LYONSCHOOL, Data.HIGHSCHOOL, Data.CONFERENCE, Data.WIFI, Data.SAFEGRAPH]
    networks = [Network.TEMPORAL]
    for data in datasets:
        G, temp_G, T, all_nodes = load_data(data, t_split=False)
        
        for ntw in networks:
        # print(number_of_connected_components(temp_G))
            # static_G, temp= load_contact_network(ntw, G, temp_G, 0, 0)
            # print(static_G)
            print(data, maximum_node_degree(temp_G, temporal=True))