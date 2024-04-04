from audioop import reverse
import networkx as nx
import numpy as np
import os
import pandas as pd
import datetime
# import math
import itertools
# import gzip
from scipy import sparse
import scipy
import json
import random
# import resource
import time
cur_dir = os.path.dirname(__file__)

def combine_two_data(file1, file2):
    case_path1 = os.path.join(cur_dir, "data", "safegraph", file1)
    data1 = pd.read_csv(case_path1, compression='gzip')
    print(data1.shape)
    print(data1)
    case_path2 = os.path.join(cur_dir, "data", "safegraph", file2)
    data2 = pd.read_csv(case_path2, compression='gzip')
    print(data2.shape)
    print(data2)
    data = data1.append(data2, ignore_index = True)
    print(data.shape)
    print(data)
    filename = os.path.join(cur_dir, "data", "safegraph", "weekly_normalization_stats_Mar_Sep.csv.gz")
    data.to_csv(filename, index=False, compression='gzip')
    return



def read_safegraph_case_data(visits_data_file):
    cur_dir = os.getcwd()

    attributes=['placekey','date_range_start','date_range_end','visitor_home_cbgs']    
    case_path = os.path.join("~/CopenhagenStudy", "data", "safegraph", visits_data_file)
    visit_data = pd.read_csv(case_path, compression='gzip', usecols=attributes)

    visit_data['date_range_start'] = pd.to_datetime(visit_data['date_range_start'])
    visit_data['date_range_end'] = pd.to_datetime(visit_data['date_range_end'])
    visit_data = visit_data.sort_values(by=['date_range_start'])

    home_path = os.path.join("~/CopenhagenStudy", "data","safegraph", "Canada-DA.csv")
    home_attr = ['Geographic code', 'Province / territory, english', 'Population, 2016']
    home_data = pd.read_csv(home_path, usecols=home_attr)
    
    home_data = home_data[(home_data['Province / territory, english']=='Quebec')][[
                            'Geographic code', 'Population, 2016']]
    home_data = home_data.reset_index(drop=True)

    
    # home_path = os.path.join(cur_dir, "data", "safegraph", home_data_file)
    # home_attr = ['date_range_start','census_block_group','number_devices_residing']
    # home_data = pd.read_csv(home_path, compression='gzip', usecols=home_attr)

    # home_data['date_range_start'] = pd.to_datetime(home_data['date_range_start'])
    # home_data = home_data.sort_values(by=['date_range_start'])

    # delete rows with empty visitor_home_cbgs
    visit_data = visit_data[(visit_data["visitor_home_cbgs"]!="{}")]
    visit_data = visit_data.reset_index(drop=True)
    print(visit_data.shape)
    
    weekly_dates = visit_data["date_range_start"].unique()

    poi_dict = {}
    cbg_dict = {}
    poi_i = 0
    cbg_i = 0

    POI_num = []
    CBG_num = []
    VISIT_num = []
    population_list = {}
    START_DATE = datetime.datetime.strptime("2020-03-09", "%Y-%m-%d").date()
    END_DATE = datetime.datetime.strptime("2020-08-31", "%Y-%m-%d").date()
    # # creating visit matrix - rows: CBG - column: POI
    for week_start in weekly_dates:
        POI_num = []
        CBG_num = []
        VISIT_num = []
        print(week_start)
        # print(visit_data)
        for i, poi_data in visit_data.iterrows():
            if poi_data["date_range_start"] == week_start:
                poi_visits = json.loads(poi_data["visitor_home_cbgs"])
                poi_name = poi_data["placekey"]
                if poi_name not in poi_dict:
                    poi_dict[poi_name] = poi_i
                    poi_i +=1
                for cbg_area, num_visits in poi_visits.items():
                    sp = cbg_area.split(":")
                    
                    if sp[0] == 'CA':
                        cbg_area = int(sp[1])
                        ind = (np.where( home_data == int(cbg_area)))[0]
                        # population = int(home_data['Population, 2016'].iloc[int(ind)])
                        if (ind.shape)[0] != 0 and not np.isnan(home_data['Population, 2016'].iloc[int(ind)]):
                            if cbg_area not in cbg_dict:
                                cbg_dict[cbg_area] = cbg_i
                                population_list[cbg_area]=int(home_data['Population, 2016'].iloc[int(ind)])
                                cbg_i += 1

                            POI_num.append(poi_dict[poi_name])
                            CBG_num.append((cbg_dict[cbg_area]))
                            VISIT_num.append(num_visits) 
        mtr = sparse.csc_matrix((VISIT_num, (CBG_num, POI_num)))
        print(mtr.shape)
        print(mtr.count_nonzero())
        print(max(VISIT_num))
        print(sum(VISIT_num))
        print(mtr.max())
        print(mtr.sum())
        # print(len(CBG_num))
        # print(len(population_list))
        # print(len(cbg_dict))
        # population_flow = mtr * mtr.transpose()
        

        # save_path = os.path.join(cur_dir, "data", "safegraph", "mobility_matrix_2", "POI_visits_"+str(week_start.date()))
        # sparse.save_npz(save_path, mtr)

    # assert len(POI_num) == len(CBG_num)
    # assert len(CBG_num) == len(VISIT_num)

    # cbg_dict = dict(sorted(cbg_dict.items()))
    # population_list = dict(sorted(population_list.items()))
    # cbg = pd.DataFrame(cbg_dict.items(), columns=['DA', 'Value'])
    # cbg["population"] = population_list.values()
    # print(cbg)
    # filename = os.path.join(cur_dir, "data", "safegraph", "quebec_DA.csv.gz")
    # cbg.to_csv(filename, index=False, compression='gzip')

    # print(cbg_dict)
    # cbg_json = json.dumps(cbg_dict)
    # cbg_path = "data/safegraph/cbg_dict.json"
    # f = open(cbg_path,"w")
    # f.write(cbg_json)
    # f.close()

    # poi_json = json.dumps(poi_dict)
    # poi_path = "data/safegraph/poi_dict.json"
    # f = open(poi_path,"w")
    # f.write(poi_json)
    # f.close()
    return 

# read_safegraph_case_data("visit_data_Mar_to_Sep.csv.gz")

def plot_safegraph_population():
    from collections import Counter
    import matplotlib.pyplot as plt
    data_path = os.path.join(cur_dir, "data", "safegraph", "safegraph_cbg_data.csv.gz")
    attributes = ["date_range_start", "census_block_group", "number_devices_residing"]
    home_panel = pd.read_csv(data_path, compression='gzip', usecols=attributes)
    home_panel = home_panel[((home_panel["date_range_start"] == "2020-03-09T00:00:00-04:00") &
                            (home_panel["number_devices_residing"]>=20))]
    population = home_panel["number_devices_residing"]
    cbg = list(range(home_panel.shape[0]))
    a = dict(Counter(population))
    a = dict(sorted(a.items(), key=lambda item: item[1]))
    a1=a.keys()
    a2=a.values()
    fig = plt.figure(facecolor='w', figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.scatter(a1, a2, color='blue')
    ax.scatter(cbg, population)
    ax.set_xlabel('number of contact with specific frequency', fontsize=16)
    ax.set_ylabel('Edge Frequency', fontsize=16)
    plt.show()
    plt.savefig("population_demographics.png")

def create_complete_home_panel():
    data_path = os.path.join("~/CopenhagenStudy", "data", "safegraph", "safegraph_cbg_data.csv.gz")
    home_panel = pd.read_csv(data_path, compression='gzip', names = ["time", "DA", "population"], header=0)
    home_panel = home_panel[(home_panel["population"] > 0)]

    data_path_real_population = os.path.join("~/CopenhagenStudy", "data", "safegraph", "quebec_DA.csv.gz")
    real_population = pd.read_csv(data_path_real_population, compression = 'gzip', usecols=["DA", "Value", "population"])
    print(home_panel)
    # date_list = home_panel["time"].unique()
    date_list = ['2020-03-09', '2020-03-16', '2020-03-23', '2020-03-30'
            , '2020-04-06', '2020-04-13', '2020-04-20', '2020-04-27'
            , '2020-05-04', '2020-05-11', '2020-05-18', '2020-05-25'
            , '2020-06-01', '2020-06-08', '2020-06-15', '2020-06-22'
            , '2020-06-29', '2020-07-06', '2020-07-13', '2020-07-20'
            , '2020-07-27', '2020-08-03', '2020-08-10', '2020-08-17'
            , '2020-08-24', '2020-08-31', '2020-09-07', '2020-09-14'
            , '2020-09-21']
    # print(home_panel)
    cbg_dict = {}
    cbg_population_dict = dict(zip(real_population["DA"], real_population["population"]))
    keys = [k for k, v in cbg_population_dict.items() if v == 0]
    print(keys)
    print("sg",len(cbg_population_dict))
    for week in date_list:
        # print(week)
        home_panel_week = home_panel[(home_panel["time"]==week)]
        for i, data in home_panel_week.iterrows():
            if data["DA"] not in cbg_dict or cbg_dict[data["DA"]] < data["population"]:
                cbg_dict[data["DA"]] = data["population"]
    print("dem",len(cbg_dict))
    cbg_dict = {k: v for k, v in cbg_dict.items() if k in cbg_population_dict.keys()}
    cbg_dict = dict(sorted(cbg_dict.items()))
    cbg_population_dict = dict(sorted(cbg_population_dict.items()))

    for i in keys:
        print(cbg_population_dict[i])
        cbg_population_dict[i] = cbg_dict[i] * 5
        print(cbg_population_dict[i], cbg_dict[i])
    print("dem",len(cbg_dict))
    print("sg",len(cbg_population_dict))
    cbg = pd.DataFrame(cbg_dict.items(), columns=['DA', 'SG_population'])
    cbg[ "real_population"] = pd.Series(cbg_population_dict.values())

    print(cbg)
    print(cbg["SG_population"].sum(axis=0))
    print(cbg.iloc[58])
    filename = os.path.join(cur_dir, "data", "safegraph", "quebec_DA_population.csv.gz")
    cbg.to_csv(filename, index=False, compression='gzip')        
    return

# create_complete_home_panel()
class safegraph:

    def __init__(self, weeks_list):
        self.weeks = weeks_list
        self.scaling_factor = 7
        self.cbg_data = None
        self.normalization = 15
        self.individual_list = []
        self.safegraph_population = []
        
        
    def poi_visits_loader(self, week):
        week = pd.to_datetime(week)
        data_path = os.path.join(cur_dir, "data", "safegraph","mobility_matrix_2", "POI_visits_"+str(week.date())+".npz")
        self.mobility_matrix = sparse.load_npz(data_path)
        self.mobility_matrix = self.mobility_matrix #* self.normalization
        return

    def mobility_data_loader(self, week):
        week = pd.to_datetime(week)
        data_path = os.path.join("~/CopenhagenStudy", "data", "safegraph","mobility_matrix", "mobility_matrix_week_"+str(week.date())+".npz")
        self.mobility_matrix = sparse.load_npz(data_path)
        return 

    def cbg_data_loader(self):
        data_path = os.path.join(cur_dir, "data", "safegraph", "quebec_DA.csv.gz")
        attributes = ["DA", "Value", "population"]
        self.cbg_data = pd.read_csv(data_path, compression = 'gzip', usecols=attributes)
        # self.cbg_dictionary = dict(zip(cbg_data["Value"] , cbg_data["DA"]))
        # print(cbg_data)
        self.population = self.cbg_data["population"].sum()
        print("total population = ", self.population)
        

    def weekly_home_panel(self):
        # for i in range(7):
        #     data_path = os.path.join(cur_dir, "data", "safegraph", "weekly_home_panel_summary-part"+str(i)+".csv.gz")
        #     attributes = ["date_range_start", "region", "iso_country_code", "census_block_group", "number_devices_residing"]
        #     home_panel = pd.read_csv(data_path, compression='gzip', usecols=attributes)
        #     # print(home_panel)
        #     home_panel = home_panel[((home_panel["region"]=='qc') & 
        #                                     (home_panel["iso_country_code"] == 'CA'))][[
        #                                         "date_range_start", "census_block_group", "number_devices_residing"]]
        #     print(home_panel.shape[0])
        #     # print(home_panel["date_range_start"].unique())
        #     if home_panel.shape[0] != 0:
        #         if i == 0:
        #             home_panel_summary = home_panel
        #             # print(self.home_panel)
        #         else:
        #             result = [home_panel_summary, home_panel]
        #             home_panel_summary = pd.concat(result, ignore_index = True)
        # print(home_panel_summary)
        # new_DA = []
        # new_time = []
        # for i, line in home_panel_summary.iterrows():
        #     cbg = line["census_block_group"].split(":")
        #     new_DA.append(cbg[1])

        #     date_only = line["date_range_start"].split("T")
        #     new_time.append(date_only[0])
        # home_panel_summary["census_block_group"] = pd.Series(new_DA)
        # home_panel_summary["date_range_start"] = pd.Series(new_time)
        # home_panel_summary.to_csv("~/CopenhagenStudy/data/safegraph/safegraph_cbg_data.csv.gz", index=False, compression='gzip')
        # data_path = os.path.join("~/CopenhagenStudy", "data", "safegraph", "safegraph_cbg_data.csv.gz")
        data_path = os.path.join("~/CopenhagenStudy", "data", "safegraph", "quebec_DA_population.csv.gz")
        self.home_panel = pd.read_csv(data_path, compression='gzip', usecols = ["DA", "SG_population", "real_population"])
        print("Real population : ", self.home_panel["real_population"].sum(axis = 0))
        print("SG population : ", self.home_panel["SG_population"].sum(axis = 0))
        # self.normalization = self.home_panel["real_population"].sum(axis = 0) / self.home_panel["SG_population"].sum(axis = 0)
        # print("normalization stat : ", self.normalization)

    def cbg_network_generator(self):
        start = time.time()
        G = nx.Graph()
        cbg_data = self.home_panel
        cbg_data = cbg_data.reset_index(drop=True)
        # self.population_list = cbg_data["SG_population"]
        print("number of CBGs : ", cbg_data.shape[0])
        
        G_random = []
        for i, cbg_i in cbg_data.iterrows():
            # if i<3000:    
                population = int(cbg_i ["SG_population"])
                p = 10 / population
                G_random.append(nx.fast_gnp_random_graph(population, p))
        G = nx.disjoint_union_all(G_random)
        print("ave degree : ", G.number_of_edges() * 2 / G.number_of_nodes())
        print("before connection : ", G)
        end = time.time()
        print("  - time elapsed: ", end - start)
        self.initial_graph = G
    
    def using_tocoo_izip(self):
        cx = self.mobility_matrix.tocoo()
        # print("sum",self.mobility_matrix.sum())  
        poi_label = 0 
        new_nodes = [] 
        weekly_G = self.initial_graph.copy()
        self.population_list = dict(zip(self.cbg_data["Value"], self.home_panel["SG_population"]))
        self.population_list = dict(sorted(self.population_list.items()))
        self.population_list = list(self.population_list.values())
        for i,j,v in zip(cx.row, cx.col, cx.data):
            # v = v * self.normalization
            # if j<8:
                # choose v random from numbers in nodes of graph i
                if poi_label != j:
                    list_graphs = itertools.combinations(range(0,len(new_nodes)), 2)
                    for graph_conn in list_graphs:
                        cbg_1 = new_nodes[graph_conn[0]]
                        cbg_2 = new_nodes[graph_conn[1]]
                        Edges = list(itertools.product(cbg_1, cbg_2))
                        Edges = random.sample(Edges, int(len(Edges)/7))
                        weekly_G.add_edges_from(x for x in Edges)
                    poi_label = j
                    new_nodes = []
                
                if i == 0:
                    lower_bound = 0
                else:
                    lower_bound = sum(self.population_list [0:i])
                upper_bound = self.population_list[i] + lower_bound
                # print(i,j, v,lower_bound, upper_bound)
                if v > (upper_bound - lower_bound):
                    v = upper_bound - lower_bound - 1
                new_nodes.append(random.sample(range(lower_bound, upper_bound),v))  
        print("after connection : ", weekly_G)      
        return weekly_G

    def run(self):
        temporal_graph = []
        static_G = []
        self.cbg_data_loader()
        self.weekly_home_panel()
        self.cbg_network_generator()
        all_edges = 0
        for i, week in enumerate(self.weeks):
            print("week", week)
            self.poi_visits_loader(week)
            temp_G = self.using_tocoo_izip()
            all_edges += temp_G.number_of_edges()
            nx.write_edgelist(temp_G, "./data/safegraph/temporal_graphs_d10/week_"+str(i)+".edgelist", data=False)
            # temporal_graph.append(temp_G)
            # static_G.append(temp_G)

        # static_G = nx.compose_all(static_G)
        print("all edges",all_edges)
        print(static_G)
        return temporal_graph, static_G

    def normalization_stats(self):
        data_path = os.path.join(cur_dir, "data", "safegraph", "weekly_normalization_stats_Mar_Sep.csv.gz")
        attributes = ["year","month","day","region", "iso_country_code", "total_visits","total_devices_seen"]
        normalization = pd.read_csv(data_path, compression = 'gzip', usecols=attributes)
        normalization = normalization[((normalization["region"]=='qc') & 
                                            (normalization["iso_country_code"] == 'CA'))][[
                                                "year","month","day", "total_visits", "total_devices_seen"]]             
        normalization = normalization.reset_index(drop = True)
        print(normalization)
        week_count = 0
        for t_w in self.weeks:
            seen_devices_num = 0
            day_count = 0
            # first_day = datetime.datetime.strptime(t_w, "%Y-%m-%d").day
            # print(np.where( normalization == first_day))
            # ind_first_day = int((np.where( normalization == first_day))[0])
            for i in range(7):
                seen_devices_num += normalization["total_devices_seen"].iloc[day_count + i]
            total_seen_devices = int(seen_devices_num / 7)
            self.safegraph_population.append(total_seen_devices)
        #     # self.normalization.append(self.population / total_seen_devices)
        #     self.normalization += self.population/total_seen_devices
            week_count += 1
        # self.normalization /= len(self.weeks)
        # print("normalization factor for Quebec = ", self.normalization)
        print(self.safegraph_population)
        return 
        data_path = os.path.join(cur_dir, "data", "safegraph", "weekly_normalization_stats_Mar_Sep.csv.gz")
        attributes = ["year","month","day","region", "iso_country_code", "total_visits","total_devices_seen"]
        normalization = pd.read_csv(data_path, compression = 'gzip', usecols=attributes)
        normalization = normalization[((normalization["region"]=='qc') & 
                                            (normalization["iso_country_code"] == 'CA'))][[
                                                "year","month","day", "total_visits", "total_devices_seen"]]             
        normalization = normalization.reset_index(drop = True)
        print(normalization)
        week_count = 0
        for t_w in self.weeks:
            seen_devices_num = 0
            day_count = 0
            # first_day = datetime.datetime.strptime(t_w, "%Y-%m-%d").day
            # print(np.where( normalization == first_day))
            # ind_first_day = int((np.where( normalization == first_day))[0])
            for i in range(7):
                seen_devices_num += normalization["total_devices_seen"].iloc[day_count + i]
            total_seen_devices = int(seen_devices_num / 7)
            self.safegraph_population.append(total_seen_devices)
        #     # self.normalization.append(self.population / total_seen_devices)
        #     self.normalization += self.population/total_seen_devices
            week_count += 1
        # self.normalization /= len(self.weeks)
        # print("normalization factor for Quebec = ", self.normalization)
        print(self.safegraph_population)
        return 

def create_bi_cbg_graphs(weeks):
    bi_cbg_cum_flow = []
    bi_cbg_flow = []
    bi_cbg_list = []
    new_nodes = {}
    for n_week, week in enumerate(weeks):
        print(week)
        poi_label = 0 
        week = pd.to_datetime(week)
        data_path = os.path.join(cur_dir, "data", "safegraph","mobility_matrix_2", "POI_visits_"+str(week.date())+".npz")
        mobility_matrix = sparse.load_npz(data_path)
        print(mobility_matrix.count_nonzero())
        cx = mobility_matrix.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
                if poi_label != j:
                    list_cbg_conn = list(itertools.combinations(new_nodes.keys(), 2))
                    for graph_conn in list_cbg_conn:
                        
                        cbg_1 = graph_conn[0]
                        cbg_2 = graph_conn[1]
                        if graph_conn in bi_cbg_list:
                            k = list_cbg_conn.index(graph_conn)
                            bi_cbg_cum_flow[k] += (new_nodes[cbg_1] * new_nodes[cbg_2])
                            if len(bi_cbg_flow[k]) < n_week+1:
                                bi_cbg_flow[k].append(1)
                            else:
                                bi_cbg_flow[k][n_week] += 1
                        else:
                            bi_cbg_cum_flow.append(new_nodes[cbg_1] * new_nodes[cbg_2])
                            bi_cbg_flow.append([1])
                            bi_cbg_list.append(graph_conn)
                    poi_label = j
                    new_nodes = {}
                new_nodes[i] = v
        for k, j in enumerate(bi_cbg_flow):
            if len(j) < n_week+1:
                bi_cbg_flow[k].append(0)
                                        

    cbg = pd.DataFrame(bi_cbg_list, columns=["i", "j"])
    cbg["list"] = bi_cbg_flow
    cbg["po"] = pd.Series(bi_cbg_cum_flow)
    cbg = cbg.sort_values(by=['list'], ascending=False)
    cbg = cbg.reset_index(drop = True)
    print(cbg.iloc[0:500])
    return cbg

def create_csv_edgelist(weeks):
    csv_edges = []
    H = nx.Graph()
    for i in range(len(weeks)):
        print("Loading data for the week", weeks[i])
        data_path = os.path.join(cur_dir, "data", "safegraph/temporal_graphs", "week_"+str(i)+".edgelist")
        with open(data_path, 'r') as fin:
            for line in fin:
                values = line.split(" ")
                node_1 = int(values[0])
                node_2 = int(values[1])
                csv_edges.append((i, min(node_1, node_2), max(node_1, node_2)))
                if H.has_edge(min(node_1, node_2), max(node_1, node_2)):
                    H[min(node_1, node_2)][max(node_1, node_2)]['weight'] +=1
                else:
                    H.add_edge(min(node_1, node_2), max(node_1, node_2), weight=1)
    print(H)
    nx.write_edgelist(H, "./data/safegraph/sg_weighted_weekly_agg_d10.edgelist", data=['weight'])
    edgelist = pd.DataFrame(csv_edges, columns=["time", "user_a", "user_b"])
    
    edgelist.to_csv("~/CopenhagenStudy/data/safegraph/sg_weekly_agg_d10.csv.gz",sep=',', compression='gzip', index=False, header=False)
    

if __name__ == "__main__":
    weeks = ['2020-03-09', '2020-03-16', '2020-03-23', '2020-03-30'
            , '2020-04-06', '2020-04-13', '2020-04-20', '2020-04-27'
            , '2020-05-04', '2020-05-11', '2020-05-18', '2020-05-25'
            , '2020-06-01', '2020-06-08', '2020-06-15', '2020-06-22'
            , '2020-06-29', '2020-07-06', '2020-07-13', '2020-07-20'
            , '2020-07-27', '2020-08-03', '2020-08-10', '2020-08-17'
            , '2020-08-24', '2020-08-31',  '2020-09-07', '2020-09-14'
            , '2020-09-21']
    # weeks = ['2020-03-09', '2020-03-16', '2020-03-23']
    nn = safegraph(weeks)
    temp_G, static_G = nn.run()
    # create_csv_edgelist(weeks)