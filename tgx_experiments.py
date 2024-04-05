import tgx
from NetworkUtils import load_temporal_edgelist_csv
from model.Network import *
from model.DataPath import *
import matplotlib.pyplot as plt


def calculate_node_per_ts(graph):
    active_nodes = []
    for ts in graph:
        active_nodes.append(ts.number_of_nodes())
    return active_nodes

def calculate_edge_per_ts(graph):
    active_edges = []
    for ts in graph:
        active_edges.append(ts.number_of_edges())
    return active_edges

def plot_edge_node(dates,y1,  title):
    fig = plt.figure(facecolor='w', figsize=(9, 6))
    ax = fig.add_subplot(111)
    dates = list(range(0, dates))
    ax.plot(dates, y1, color='black', lw=2, label= 'active nodes')
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Edge Per Timestamp', fontsize=16)
    ax.tick_params(labelsize=15)
    plt.show()
    plt.savefig(title)


dataSET = [Data.SAFEGRAPH]
for data in dataSET:
    temporal_data_path = "data"+ DataPath[str(data.name)]+ "agg.csv.gz"
    temp_G = load_temporal_edgelist_csv("./"+temporal_data_path)
    nodes = calculate_node_per_ts(temp_G)
    plot_edge_node(len(temp_G), nodes, "./graph/"+ResultPath[str(data.name)]+"/node_per_ts.png")    