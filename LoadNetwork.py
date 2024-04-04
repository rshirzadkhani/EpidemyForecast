from NetworkUtils import *
from model.Safegraph_analyse import safegraph
from model.Network import Network, Data
from model.DataPath import DataPath
import random
from contact_network_utils import matching_degree_top_edge, matching_degree_top_node, network_density, node_degree_count, MST
import networkx as nx
cur_dir = os.path.dirname(__file__)


def load_data(data, t_split=0):
    print("Loading "+data.name+" dataset...")
    static_data_path = "data"+ DataPath[str(data.name)]+ "weighted_edgelist_agg.edgelist"
    temporal_data_path = "data"+ DataPath[str(data.name)]+ "agg.csv.gz"
    G = nx.read_weighted_edgelist(os.path.join(cur_dir, static_data_path), nodetype=int)
    print("Loading temporal graphs...")
    temp_G = load_temporal_edgelist_csv("./"+temporal_data_path)


    T = len(temp_G)
    dates = list(range(0, T+1))
    all_nodes = G.number_of_nodes()
    
    if t_split > 0:
        # t_split = np.round(0.25 * T)
        print("Creating a split for data at time {}.".format(t_split))
        G, temp_G = create_prediction_edgelist("./"+temporal_data_path, t_split)
        print("Splitted network:", G)
    return G, temp_G, T, all_nodes


def load_contact_network(network, G, temp_G, seed, t_split):
    random.seed(seed)
    np.random.seed(seed)
    all_nodes = G.number_of_nodes()
    T = len(temp_G)
    degree = network_density(G, temp_G, T, all_nodes)
    print("temporal degree for split section:", degree)
    if network.name == Network.STATIC.name:
        temporal_G = None
        static_G = G

    elif network.name == Network.MST_W_MATCH.name:
        temporal_G = None
        all_nodes = G.number_of_nodes()
        degree = network_density(G, temp_G, len(temp_G), all_nodes)
        init_G = MST(G)
        static_G = matching_degree_top_edge(G, init_G, degree)


    elif network.name == Network.MST_D_MATCH.name:
        temporal_G = None
        all_nodes = G.number_of_nodes()
        degree = network_density(G, temp_G, len(temp_G), all_nodes)
        init_G = MST(G)
        node_list = node_degree_count(temp_G, G)
        static_G = matching_degree_top_node(node_list, G, init_G, degree)


    elif network.name == Network.TEMPORAL.name:
        temporal_G = temp_G
        static_G = G
        
    elif network.name == Network.ER.name:
        # degree = network_density(G, None, len(temp_G), all_nodes)
        n = G.number_of_nodes()
        p = degree / (n-1)
        static_G = nx.fast_gnp_random_graph(n, p)
        temporal_G = None

    elif network.name == Network.BA.name:
        temporal_G = None
        p = 1
        for m in np.arange(2,100, 1):
            syn_G = nx.dual_barabasi_albert_graph(all_nodes, m1=m, m2=1, p=1)
            g_degree = network_density(syn_G, temporal_G, len(temp_G), all_nodes)
            if g_degree >= degree:
                best_m = m
                print(best_m)
                break
        static_G = nx.barabasi_albert_graph(all_nodes, best_m)

    elif network.name == Network.STANDARD_GRAPH.name:
        temporal_G = []
        # degree = network_density(G, None, len(temp_G), all_nodes)
        for x in range(len(temp_G)):
            if (degree * all_nodes) % 2 != 0:
                all_nodes += 1
            new_graph = nx.random_regular_graph(degree, all_nodes, seed=np.random)
            print(new_graph)
            temporal_G.append(new_graph)
        static_G = nx.compose_all(temporal_G)


    print("Number of Nodes:", static_G.number_of_nodes(), "|| Edges:", static_G.number_of_edges())
    return static_G, temporal_G