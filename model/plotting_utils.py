import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import numpy as np
import collections
import matplotlib.cm as cm
import os
from scipy.ndimage import gaussian_filter1d

# from model.Constants import SIR_COMPARTMENTS, MOBILITY_TYPE_SYN, NETWORK_TYPES, SEIR_COMPARTMENTS, \
#      COVID_MODEL_NAMES
# from model.utils import get_sir_by_scope

cur_dir = os.path.dirname(__file__)
        
def plot_for_contact_network(x, sir_all, lower, upper, filename, labels):
    colors = ["#8B2323","#1f77b4", "#ff7f0e" , "#2E2E2E"] #real "#1a9850" blue:#67a9cf orange: "#fe9929" brown:"#8B2323" green:#2ca02c
    lnstyle = ['solid','dashed', 'dashed','solid']
    # colors = ["#fe9929","#67a9cf","#8B2323" ,"#02008a" ,"#2E2E2E"] #synthetic
    
    # lnstyle = ['solid','solid', 'dashed','dashed', 'solid']
    # print(len(sir_all))
    title=None
    fig = plt.figure(facecolor='w', figsize=(12,9))
    ax = fig.add_subplot(111)
    for idx, label in enumerate(labels):
        # print(idx)
        # ax.plot(x, sir_all[idx, 0, :, 0], color=colors[idx % len(colors)], linestyle = lnstyle[idx], lw=3.5, label=label)
        ax.plot(x, sir_all[idx], color=colors[idx % len(colors)], linestyle = lnstyle[idx], lw=3.5, label=label)
        ax.fill_between(x, (lower[idx]), (upper[idx]), color=colors[idx % len(colors)], alpha=.15)
        # ax.fill_between(x, y_lower[idx, 0, :, 0], y_upper[idx, 0, :, 0], color='k', alpha=.1)
    ax.set_xlabel("Time", fontsize=40)
    ax.set_ylabel('Infections', fontsize=40)
    ax.tick_params(labelsize=40)
    ax.grid(b=False)
    # legend = ax.legend(fontsize=25)
    # legend.get_frame().set_alpha(0.95)
    if title is not None:
        ax.set_title(title, fontsize=20)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)

def plot_degree_distribution(degree_freq, node_num, labels, data, network):
    filepath = "/graph/{}/degree_dist/{}"
    fig = plt.figure(facecolor='w', figsize=(9,6))
    ax = fig.add_subplot(111)
    for idx, ntw_degree_freq in enumerate(degree_freq):
        degree_prob=[i/node_num for i in ntw_degree_freq]
        degrees = range(len(ntw_degree_freq))
        # degree_prob=[i/G.number_of_nodes() for i in degree_freq]
        ax.scatter(degrees, degree_prob, label=labels[idx]) 
    #z = np.polyfit(np.log(degrees), np.log(degree_freq), 1)
    #fitted = [np.exp(np.log(d) * z[0] + z[1]) for d in degrees]
    #plt.plot(degrees, fitted)
    #plt.plot(np.unique(degree_sequence, return_counts=True), "b-", marker="o")
    plt.yscale("log")
    plt.xscale("log")
    legend = ax.legend(fontsize=18)
    legend.get_frame().set_alpha(0.5)
    plt.xlabel("degree")
    plt.ylabel("probability")
    plt.savefig(filepath)


def plot_edge_node(dates,y1,  title):
    fig = plt.figure(facecolor='w', figsize=(9, 6))
    ax = fig.add_subplot(111)
    dates = list(range(0, len(dates)-1))
    ax.plot(dates, y1, color='black', lw=2, label= 'active nodes')
    # ax.plot(dates, y2, alpha=0.5, color='blue', lw=2, label = 'all nodes')
    # ax.plot(dates, y2,  "r--", alpha=0.5, lw=2, label="contacts with distance<2m")
    # ax.axhline(y=np.nanmean(y1), color='black', linestyle='dashed')
    # ax.axhline(y=np.nanmean(y2),  color='blue', linestyle='dashed')
    ax.axhline(y=np.nanmean(y1),  color='black', linestyle='dashed')
    ax.set_xlabel('time', fontsize=16)
    ax.set_ylabel('snapshot density', fontsize=16)
    # start, end = ax.get_ylim()
    # ax.yaxis.set_ticks(np.arange(0, end, 20))
    # ax.set_xlim(1,28)
    # ax.set_xticks(np.arange(0,29,7))
    # ax.set_yticks(np.arange(0,601,100))
    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.tick_params(labelsize=15)
    # ax.legend(fontsize=14)
    # if title is not None:
    #     plt.title(title, fontsize=16)
    plt.show()
    plt.savefig(title)