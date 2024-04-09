
from ast import Break
from sqlite3 import Timestamp
import networkx as nx
import numpy as np
from model.SEIR import SEIR
from model.Network import Network
import os
import pandas as pd
import datetime
import math
import itertools
import random
import gzip
import time
from scipy import sparse
import json
cur_dir = os.path.dirname(__file__)


def load_temporal_edgelist(fname):
    data_path = os.path.join(cur_dir, fname)
    edgelist = open(data_path, "r")
    lines = list(edgelist.readlines())
    edgelist.close()
    cur_t = 0

    '''
    t u v
    '''
    G_times = []
    G = nx.Graph()

    yy = 0
    for i in range(0, len(lines)):
        if (i == 0):
            continue
        line = lines[i]
        values = line.split(",")
        t = int(values[0])
        u = int(values[1])
        v = int(values[2])
        #start a new graph with a new date
        if (t != cur_t):
            G_times.append(G)   #append old graph
            # xx += G.number_of_nodes()
            yy += G.number_of_edges()
            G = nx.Graph()  #create new graph
            cur_t = t 
        G.add_edge(u, v) 

        # p1 = np.where((edge_list == u))
        # p2 = np.where((edge_list == v))
        # uv = list(set(p1[0]) & set(p2[0]))

        # if len(uv)==1 and edge_freq[uv[0]]<(t1+1):
        #     edge_freq[uv[0]] += 1
        # elif len(uv)==0:
        #     edge_list = np.append(edge_list, [[u , v]], axis=0)
        #     edge_freq.append(1)
    G_times.append(G)
    print("Total number of contacts: ", yy)
    print ("Maximum time stamp: " + str(len(G_times)))
    return G_times

def load_temporal_edgelist_csv(fname):
    data_path = os.path.join(cur_dir, fname)
    # data = pd.read_csv(data_path, compression='gzip', names = ["time", "user_a", "user_b"])
    edgelist = gzip.open(data_path, "rt")
    lines = list(edgelist.readlines())
    edgelist.close()

    cur_t = 0
    G_times = []
    G = nx.Graph()
    yy = 0
    for i in range(0, len(lines)):
        if (i == 0):
            continue
        line = lines[i]
        values = line.split(",")
        t = int(values[0])
        u = int(values[1])
        v = int(values[2])

        #start a new graph with a new date
        if (t != cur_t):
            G_times.append(G)   #append old graph
            # xx += G.number_of_nodes()
            yy += G.number_of_edges()
            G = nx.Graph()  #create new graph
            cur_t = t 
        G.add_edge(u, v) 
    G_times.append(G)
    print("Total number of contacts: ", yy)
    print ("Maximum time stamp: " + str(len(G_times)))
    return G_times

def create_prediction_edgelist(fname, t_split):
    data_path = os.path.join(cur_dir, fname)
    edgelist = gzip.open(data_path, "rt")
    lines = list(edgelist.readlines())
    edgelist.close()
    G_static = nx.Graph()
    cur_t = 0
    temp_edges = {}
    G_times = []
    G = nx.Graph()
    for i in range(0, len(lines)):
        if (i == 0):
            continue
        line = lines[i]
        values = line.split(",")
        t = int(values[0])
        node_1 = min(int(values[1]), int(values[2]))
        node_2 = max(int(values[1]), int(values[2]))
        
        if t <= t_split:
            if (node_1, node_2) in temp_edges and t not in temp_edges[(node_1, node_2)]:
                temp_edges[(node_1, node_2)].append(t)
                G_static[node_1][node_2]['weight'] +=1
            elif (node_1, node_2) not in temp_edges:
                temp_edges[(node_1, node_2)] = [t]
                G_static.add_edge(node_1, node_2, weight=1)

            if (t != cur_t):
                G_times.append(G) 
                G = nx.Graph()  
                cur_t = t 
            G.add_edge(node_1, node_2) 
        else:
            active_nodes = G_static.nodes()
            if (t != cur_t):
                G_times.append(G) 
                G = nx.Graph()  
                cur_t = t 
            if node_1 in active_nodes and node_2 in active_nodes:
                G.add_edge(node_1, node_2) 
    G_times.append(G)
    print(len(G_times))
    return G_static, G_times


def aggregate_data(data):
    data_path = os.path.join(cur_dir,"data", data)
    data = pd.read_csv(data_path)
    t_max = data['timestamp'].max()
    hour = 24
    min_5 = 12 
    hours = int( t_max / (min_5 * 300)) + 1 
    for i in range(0, hours):
        for j in range(i*min_5*300, (i+1)*min_5*300, 300):
            data=data.replace({'timestamp': j}, i)

    data.to_csv("~/CopenhagenStudy/full_clean_data_hourly_agg.csv", index=False)
    return 


def create_edgelists():
    """this function creates all the edge lists based 
    on a 5 min resolution and save them in data folder"""
    data=pd.read_csv("~/EpidemyForecast/data/sfhh/sfhh_hourly_agg.edgelist", sep=',')
    print(data.shape)
    # t_max=data['timestamp'].max()
    # full_list=data[(data["rssi"] >= -80)][["timestamp","user_a","user_b"]]
    # time_list=[t for t in np.arange(0,t_max+300,300)]
    data.to_csv("~/EpidemyForecast/data/sfhh/sfhh_hourly_agg.csv.gz",sep=',', index=False, header=False)
    # for i in time_list:
    #     data_p=data[(data["timestamp"]==i)][["user_a","user_b"]]
    #     data_path = os.path.join(cur_dir,"data", "school",'school_edgelist_'+str(i)+'.edgelist')
    #     data_p.to_csv(data_path, sep=' ', index=False, header=False)
        
    # data_path = os.path.join(cur_dir,"data", 'full_school_edgelist_daily_agg.edgelist')
    # g = nx.read_edgelist(data_path, nodetype=int)
    # print(g)
    return 

def clean_data_nodes(data):
    """In this function we make sure number of nodes are consecutive numbers-
    when we delet some connections based on distance, we need to do this again"""
    
    data=pd.DataFrame(data[["timestamp" , "user_a","user_b", "rssi"]])
    data=data.reset_index(drop=True)
    editted_nodes={}    
    ii=0
    for idx in range(data.shape[0]):
        if data['user_a'][idx] not in editted_nodes:
            editted_nodes[data["user_a"][idx]]=ii
            ii+=1
        if data["user_b"][idx] not in editted_nodes:
            editted_nodes[data["user_b"][idx]]=ii
            ii+=1            
    data=data.replace({'user_a':editted_nodes,'user_b':editted_nodes})
    return data


def edge_count(fname):
    data_path = os.path.join(cur_dir, "data",  fname)
    edgelist = pd.read_csv(data_path, compression='gzip', sep = ',', names=["time", "user_a", "user_b"])
    print(edgelist)
    edge_list = np.array([[edgelist["user_a"][0], edgelist["user_b"][0]]])
    edge_freq = [1]
    
    number_of_days = edgelist['time'].unique()

    t_max = edgelist['time'].max()
    for t1 in range(len(number_of_days)):
        # print(t1)
        t_edgelist = edgelist[(edgelist['time']==number_of_days[t1])]
        # print(t_edgelist.shape)
        for i,line in t_edgelist.iterrows():

            u = line['user_a']
            v = line['user_b']

            p1 = np.where((edge_list == u))
            p2 = np.where((edge_list == v))
            uv = list(set(p1[0]) & set(p2[0]))
            
            if len(uv)==1 and edge_freq[uv[0]]<(t1+1):
                edge_freq[uv[0]] += 1
            elif len(uv)==0:
                edge_list = np.append(edge_list, [[u , v]], axis=0)
                edge_freq.append(1)

    edgelist = pd.DataFrame(edge_list, columns=['user_a','user_b'])
    edgelist['freq'] = pd.Series(edge_freq) 
    edgelist.to_csv("~/CopenhagenStudy/data/safegraph/sg_weighted_weekly_agg.edgelist",sep=' ', index=False, header=False)
    return 

# edge_count("./wifi/wifi_edgelist_weekly_agg_2009.edgelist")
# edge_count("./Gallery/gallery_edgelist_daily_agg.edgelist")
# edge_count("./highschool/highschool_2012_edgelist_hourly_agg.edgelist")

def wifi_edgelist_creator(data, start_date, end_date):
    data_path = os.path.join(cur_dir,"data", data)
    # dates = ["8/14/2005", "12/17/2006",#39978
    #         "1/6/2008","8/24/2008", #39799
    #         "1/1/2009", "6/29/2009"] #39989
    # start_end_dates = ["1/1/2007", "10/30/2007",
    #         "1/1/2008","10/30/2008", 
    #         "1/1/2009", "10/30/2009"] 
    data = pd.read_csv(data_path, compression='gzip',names=["user", "node", "login", "logout"])
    START_DATE = datetime.datetime.strptime(start_date, "%m/%d/%Y").date()
    END_DATE = datetime.datetime.strptime(end_date, "%m/%d/%Y").date()
    date_list = pd.date_range(start = START_DATE, end = END_DATE, freq="W")
    print("number of weeks : ", len(date_list))

    data ["login"] = pd.to_datetime(data["login"])
    data ["logout"] = pd.to_datetime(data["logout"])
    # data = data.drop('conn_id', axis=1)
    edit_n={}
    # editted_poi={}    
    # ii=0
    # ij=0
    # for idx in range(data.shape[0]):
    #     if data['node_id'][idx] not in editted_nodes:
    #         editted_nodes[data["node_id"][idx]]=ii
    #         ii+=1
        # if data["node_id"][idx]not in editted_poi:
        #     editted_poi[data["node_id"][idx]]=ij
        #     ij+=1
    # data=data.replace({'node_id':editted_nodes})
    # data ["login"]= [datetime.datetime.strptime(s.split(" ")[0], '%Y-%m-%d') for s in data['login']]
    # data ["logout"]= [data["login"][i] if len(str(s))==3 else datetime.datetime.strptime(s.split(" ")[0], '%Y-%m-%d')  for i,s in enumerate(data['logout'])]
    # duration_list = [(data["logout"][i]-data["login"][i]).days for i in range(data.shape[0])]
    G_times = []
    yy = 0
    ii = 0
    ij = 0
    H = nx.Graph()
    community_list = []
    # data.to_csv("~/CopenhagenStudy/data/wifi_raw_data_2.csv",sep=',', index=False)
    for day_num, day1 in enumerate(date_list):
        G = nx.Graph()
        day7 = day1 + datetime.timedelta(days=6)
        active_nodes = data[(((data["login"] >= day1) & (data["login"] <= day7))            # List of active nodes in a week
                                | ((data["logout"] >= day1) & (data["logout"] <= day7))
                                | ((data["login"] <= day1) & (data["logout"] >= day7)))][[
                                    "user","node"]]
        node_numbers = active_nodes["node"].unique()

        # only monitoring nodes that have been active in the first 20 days
        unique_active_users = active_nodes["user"].unique()
        if day_num < 20:
            for n in unique_active_users:
                community_list.append(n)
        else:
            unique_active_users = [n for n in unique_active_users if n in community_list]
        print("Week: ", day_num)


        for i, nn in enumerate(node_numbers):
            each_node_connections = active_nodes[(active_nodes["node"]==nn)][["user"]]
            unique_nodes = each_node_connections["user"].unique()
            # print(type(unique_nodes))
            # print(unique_nodes.shape)
            unique_nodes = [n for n in unique_nodes if n in unique_active_users]
            if len(unique_nodes)>1:
                daily_contacts = itertools.combinations(unique_nodes, 2)

                for dc in daily_contacts:
                    if dc[0] not in edit_n:
                        edit_n[dc[0]]=ii
                        ii+=1
                    if dc[1] not in edit_n:
                        edit_n[dc[1]]=ii
                        ii+=1
                    G_times.append((day_num, edit_n[dc[0]], edit_n[dc[1]]))
                    # G.add_edge(edit_n[dc[0]], edit_n[dc[1]])
                    if H.has_edge(edit_n[dc[0]], edit_n[dc[1]]):
                        H[edit_n[dc[0]]][edit_n[dc[1]]]['weight'] +=1
                    else:
                        H.add_edge(edit_n[dc[0]], edit_n[dc[1]], weight=1)
        # G_times.append(G)
        # print(G_times[0].edges())
        # print(G)
        # yy += G.number_of_edges()
    print(H)
    nx.write_edgelist(H, "./data/wifi/wifi_weighted_weekly_agg.edgelist", data=['weight'])
    wifi_edgelist = pd.DataFrame(G_times, columns=["time", "user_a", "user_b"])
    wifi_edgelist.to_csv("~/CopenhagenStudy/data/wifi/wifi_weekly_agg.csv.gz",sep=',', compression='gzip', index=False, header=False)
    print(wifi_edgelist)
    print("Total number of edges: ", yy)
    return H, G_times


def load_csv_edgelist(file):
    data_path = os.path.join("~/CopenhagenStudy","data", "conference2", file)
    edgelist = pd.read_csv(data_path, sep="\t", names=["time", "user_a", "user_b"])
    print(edgelist)
    print(edgelist.shape)
    # edgelist = edgelist.drop('res1', axis=1)
    # edgelist = edgelist.drop('res2', axis=1)
    a = []
    day_i = 0
    for i, s in edgelist.iterrows():
        # if i<10:
            tt = datetime.datetime.fromtimestamp(int(s["time"])).hour
            # print(datetime.datetime.fromtimestamp(int(s["time"])))
            # print(tt)
            if i == 0:
                a.append(0)
            elif tt == x1:
                a.append(day_i)
            else:
                a.append(day_i+1)
                day_i += 1
            x1 = tt
            # print(a)
    edgelist["time"]=pd.Series(a) 
    number_of_days = edgelist['time'].unique()
    print(len(number_of_days))
    time_dict = dict(zip(number_of_days, list(range(len(number_of_days)))))
    print(time_dict)
    edgelist = edgelist.replace({"time" : time_dict})
    print(edgelist)
    edgelist = edgelist[["time", "user_a", "user_b"]]
    edgelist.to_csv("~/CopenhagenStudy/data/conference2/conf_hourly_agg.edgelist",sep=',', index=False, header=False)
    return 


if __name__ == "__main__":
    # wifi_edgelist_creator("./wifi/wifi_raw_data_3.csv.gz","1/1/2009", "3/7/2010")
    # edge_count("safegraph/sg_weekly_agg.csv.gz")
    # load_csv_edgelist("tij_SFHH.dat_.gz")
    # load_csv_edgelist("tij_InVS15.dat_.gz")
    # load_csv_edgelist("highschool_2012.csv.gz")
    # load_csv_edgelist("ht09_contact_list.dat.gz")
    # load_csv_edgelist("copresence-SFHH.edges")
    # print(datetime.datetime.fromtimestamp(1442))
    # print(datetime.datetime.fromtimestamp(51119))

    # edge_count("./conference2/conf_hourly_agg.edgelist")
    create_edgelists()