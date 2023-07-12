# from typing_extensions import Self
import numpy as np
import random
import networkx as nx


from model.Constants import SEIR_STATES, SEIR_COMPARTMENTS


class NetworkedSEIR:

    def __init__(self, transmissibility, sigma, symptomatic_rate, gamma, duration, weight=False, scale=1,all_nodes=0,removing_nodes=None, no_npi_nodes=set()):
        self.transmissibility = transmissibility
        # self.beta = transmissibility
        self.sigma = sigma
        self.gamma = gamma
        self.duration = duration
        self.seir = None
        self.active_infected = set()
        self.scale = scale
        self.no_npi_nodes = []
        self.transmissibility_strengths = [1, 0.8, 0.6]
        self.vary_transmissibility = False
        self.quarantine = False
        self.hubs_to_keep = set()
        self.removed_edges = []
        self.reopened = False
        self.new_seir = [[0,0,0,0]]
        self.cumulative_seir = [[0,0]]
        self.new_cases=[]
        self.weight=weight
        self.total_population = all_nodes
        self.removing_nodes = removing_nodes

    def run(self, static_graph, temporal_graph):
        self.reset(static_graph)
        for t in range(self.duration):
            if temporal_graph is not None:
                graph = temporal_graph[t]
            else:
                graph = static_graph

            graph.remove_nodes_from(self.removing_nodes)
            # self.transmissibility.append(self.beta / (graph.number_of_edges() * 2 / static_graph.number_of_nodes()))
            new_exposed = self.get_new_exposed(graph, t)
            new_infected = self.get_new_infected(t)
            new_recovered = self.get_new_recovered(t)
            self.update_seir(t, new_exposed, new_infected, new_recovered)
            self.update_new( new_exposed, new_infected, new_recovered)
            self.update_cumulative_seir(t, new_infected, new_recovered)
        # self.gephi_extract(static_graph, temporal_graph)
        return self.to_seir()


    def reset(self, graph):
        self.seir = []
        compartments = [set(), set(), set(), set(), set()]
        for x, y in graph.nodes(data=True):
            compartments[SEIR_STATES.index(y['state'])].add(x)
        self.seir.append(compartments)
        self.active_infected = self.seir[0][2].union(self.seir[0][3])


    def get_new_exposed(self, graph, t):
        susceptible = self.seir[t][0]
        # print(len(susceptible))
        contacts = []
        p1 = []
        inactivate = set()
        for i in self.active_infected:
            if i in graph.nodes:
                if set(graph.neighbors(i)).isdisjoint(susceptible):
                    continue
                    # inactivate.add(i)
                else:
                    if self.scale == 1:
                        contacts += graph.neighbors(i)
                    else:
                        neighbours = list(graph.neighbors(i))
                        neighbours_exposed = random.sample(neighbours, int(len(neighbours) * self.scale))
                        contacts += neighbours_exposed
        success = np.random.uniform(0, 1, len(contacts))  < self.transmissibility
        success_contact = list(np.extract(success, contacts))
        return set(success_contact).intersection(susceptible)

    def get_new_recovered(self, t):
        infected = self.seir[t][2]
        num_recovered = int(self.gamma * (len(infected)))
        new_recovered = set(np.random.choice(list(infected), num_recovered, replace=False))
        return new_recovered

    def get_new_infected(self, t):
        exposed = self.seir[t][1]
        num_infected = int(self.sigma * len(exposed))
        new_infected = set(np.random.choice(list(exposed), num_infected, replace=False))
        return new_infected      

    def update_seir(self, t, new_exposed, new_infected, new_recovered):
        susceptible_next = self.seir[t][0].difference(new_exposed)
        exposed_next = self.seir[t][1].union(new_exposed).difference(new_infected)
        infected_next = self.seir[t][2].union(new_infected).difference(new_recovered)
        recovered_next = self.seir[t][3].union(new_recovered)
        next_seir = [susceptible_next, exposed_next, infected_next, recovered_next]
        self.active_infected = self.active_infected.union(new_infected)\
            .difference(new_recovered)
        self.seir.append(next_seir)

    def update_cumulative_seir(self, t, new_infected, new_recovered):
        cum_infected = self.cumulative_seir[t][0]+len(new_infected)
        cum_recovered = self.cumulative_seir[t][1]+len(new_recovered)
        self.cumulative_seir.append([cum_infected,cum_recovered])

    def to_seir(self):
        seir_0 = np.zeros((self.duration+1, len(SEIR_COMPARTMENTS)))
        seir_new = np.zeros((self.duration+1, len(SEIR_COMPARTMENTS)-1))
        seir_cum = np.zeros((self.duration+1, 2))
        for t in range(self.duration+1):
            s, e, i, r = len(self.seir[t][0]), \
                         len(self.seir[t][1]), \
                         len(self.seir[t][2]), \
                         len(self.seir[t][3])
            seir_0[t] = [s, e, i, r]

            e_n, i_n, r_n = self.new_seir[t][1], \
                self.new_seir[t][2], \
                self.new_seir[t][3]
            seir_new[t] = [e_n, i_n, r_n]

            i_c = self.cumulative_seir[t][0]
            r_c = self.cumulative_seir[t][1]
            seir_cum[t] = [i_c, r_c]
        return seir_0/ self.total_population, seir_new/ self.total_population , seir_cum /self.total_population# np.expand_dims(np.sum(seir_0, axis=1), 1)        

    def update_new(self, new_exposed, new_infected, new_recovered):
        num_new_exposed = len(new_exposed)
        num_new_infected = len(new_infected)
        num_new_recovered = len(new_recovered)
        num_new_susceptable = -1*(num_new_exposed)
        new_seir = [num_new_susceptable, num_new_exposed, num_new_infected, num_new_recovered]
        self.new_seir.append(new_seir)


    def gephi_extract(self, sg, tg):
        inf = self.seir
        print(inf[3])
        for t in [1,3,5]:
            if tg is not None:
                sg = tg[t]
            disease = list(inf[t][2])
            print(disease)
            nodes = sg.nodes()
            infect = []
            for s in nodes:
                if s in disease:
                    infect.append(True)
                else:
                    infect.append(False)
            print(len(nodes))
            print(len(infect))
            inf_dict = dict(zip(nodes,infect))
            nx.set_node_attributes(sg, inf_dict, "inf")
            # nx.write_gexf(sg, "./graph/gephi/sfhh/sfhh_fulls_"+str(t)+".gexf")


    def run_npi(self, static_graph, temp_G, npis, t_apply_npis, **kwargs):
        self.reset(static_graph)
        npi_idx = 0
        if type(t_apply_npis) == int :
            t_apply_npis = [t_apply_npis]
        t_apply_npi = t_apply_npis[npi_idx]
        npi = npis[npi_idx]
        new_exposed = []
        for t in range(self.duration):
            
            edges_to_remove = []
            if temp_G is not None:
                graph = temp_G [t]
                for npi in npis:
                    if npi == 'hub':
                        edges_to_remove += self.get_edges_to_remove_hub(graph, **kwargs)
                    if npi == "distance":
                        edges_to_remove += self.get_edges_to_remove_social_distancing(graph, **kwargs)
                    if npi == "quarantine":
                        # self.quarantine = True
                        graph.remove_edges_from(self.get_remove_edges_quarantine(graph, **kwargs))
                    if npi == "mask":
                        self.vary_transmissibility = True
                        new_exposed = self.get_new_exposed_varying_transmissibility(graph, t)
                        

            else:  
                graph = static_graph            
                while t == t_apply_npi:
                    if npi == 'hub':
                        edges_to_remove += self.get_edges_to_remove_hub(graph, **kwargs)
                    elif npi == "distance":
                        edges_to_remove += self.get_edges_to_remove_social_distancing(graph, **kwargs)
                    elif npi == "quarantine":
                        self.quarantine = True
                        # graph.remove_edges_from(self.get_remove_edges_quarantine(graph, **kwargs))
                    elif npi == "mask":
                        self.vary_transmissibility = True
                        # new_exposed = self.get_new_exposed_varying_transmissibility(graph, t)
                    
                    npi_idx += 1
                    if npi_idx >= len(t_apply_npis):
                        graph.remove_edges_from(edges_to_remove)
                        break
                    t_apply_npi = t_apply_npis[npi_idx]
                    npi = npis[npi_idx]

                if self.quarantine:
                    graph.remove_edges_from(self.get_remove_edges_quarantine(graph, **kwargs))
                
                new_exposed = []
                if self.vary_transmissibility:
                    new_exposed = self.get_new_exposed_varying_transmissibility(graph, t)
            if self.vary_transmissibility is not True:
                new_exposed = self.get_new_exposed(graph, t)
            new_infected_s, new_infected_a = self.get_new_infected(t)
            new_recovered_s, new_recovered_a = self.get_new_recovered(t)
            self.update_seir(t, new_exposed, new_infected_s, new_infected_a, new_recovered_s, new_recovered_a)
            self.update_new( new_exposed, new_infected_s, new_infected_a, new_recovered_s, new_recovered_a)
        return self.to_seir()


    def get_edges_to_remove_hub(self, graph, **kwargs):
        percent_remove = kwargs['hub_p']

        if len(self.no_npi_nodes) == 0:
            degree_sequence = sorted(graph.degree, key=lambda x: x[1], reverse=True)
        else:
            degree_sequence = sorted(graph.degree(set(range(graph.order())) - self.no_npi_nodes), 
                                     key=lambda x: x[1], reverse=True)
                                     
        node, deg = zip(*degree_sequence)
        num_remove = int(np.rint(graph.number_of_nodes() * percent_remove))
        nodes_to_remove = list(node)[:num_remove]
        edges_to_remove = []
        if "p_success" in kwargs:
            p_success = kwargs['p_success']
            success = np.random.uniform(0, 1, len(nodes_to_remove)) < p_success
            success_remove = list(np.extract(success, nodes_to_remove))
            self.hubs_to_keep |= set(np.extract(~success, nodes_to_remove))
            edges_to_remove = graph.edges(success_remove)
        else:
            edges_to_remove = graph.edges(nodes_to_remove)
        return list(edges_to_remove)

    def get_edges_to_remove_social_distancing(self, graph, **kwargs):
        edges_to_remove = []

        if len(self.no_npi_nodes) == 0:
            node_degs = list(graph.degree)
        else:
            node_degs = list(graph.degree(set(range(graph.order())) - self.no_npi_nodes))
        for node, deg in node_degs:
            if "max_deg" in kwargs:
                max_deg = kwargs['max_deg']
                if node not in self.hubs_to_keep and deg > max_deg:
                    edges = np.array(list(graph.edges(node)))
                    edges = edges[~(np.isin(edges[:,1], self.hubs_to_keep))]
                    edges = [tuple(e) for e in edges]
                    edges_to_remove = edges_to_remove + random.sample(edges, len(edges)-max_deg)
            elif "distance_p" in kwargs:
                percent_remove = kwargs['distance_p']
                edges = list(graph.edges(node))
                edges_to_remove += random.sample(edges, int(np.rint(deg * percent_remove)))
        return edges_to_remove


    def get_new_exposed_varying_transmissibility1(self, graph, t):
        susceptible = self.seir[t][0]
        contacts = []
        contact_transmissibility = []
        inactivate = set()
        for i in self.active_infected:
            if set(graph.neighbors(i)).isdisjoint(susceptible):
                inactivate.add(i)
            else:
                if self.scale == 1:
                    contacts += graph.neighbors(i)
                    if i in self.no_npi_nodes:
                        contact_transmissibility += [ self.transmissibility_strengths[0] * self.transmissibility if neighbour in self.no_npi_nodes else self.transmissibility_strengths[1] * self.transmissibility for neighbour in graph.neighbors(i)]
                    else:
                        contact_transmissibility += [ self.transmissibility_strengths[1] * self.transmissibility if neighbour in self.no_npi_nodes else self.transmissibility_strengths[2] * self.transmissibility for neighbour in graph.neighbors(i)]
                else:
                    neighbours = list(graph.neighbors(i))
                    neighbours_exposed = set(random.sample(neighbours, int(len(neighbours) * self.scale)))
                    contacts += neighbours_exposed
                    if i in self.no_npi_nodes:
                        contact_transmissibility += [ self.transmissibility_strengths[0] * self.transmissibility if neighbour in self.no_npi_nodes else self.transmissibility_strengths[1] * self.transmissibility for neighbour in neighbours_exposed]
                    else:
                        contact_transmissibility += [ self.transmissibility_strengths[1] * self.transmissibility if neighbour in self.no_npi_nodes else self.transmissibility_strengths[2] * self.transmissibility for neighbour in neighbours_exposed]
                
        self.active_infected = self.active_infected.difference(inactivate)
        contact_probability = np.random.uniform(0, 1, len(contacts))
        success = np.where(contact_probability < contact_transmissibility)
        success_contact = list(np.extract(success, contacts))
        return set(success_contact).intersection(susceptible)

    def get_new_exposed_varying_transmissibility(self, graph, t):
        susceptible = self.seir[t][0]
        contacts = []
        contact_transmissibility = []
        inactivate = set()
        for i in self.active_infected:
            if i in graph.nodes:
                if set(graph.neighbors(i)).isdisjoint(susceptible):
                    inactivate.add(i)
                else:
                    if self.scale == 1:
                        contacts += graph.neighbors(i)
                        if i in self.no_npi_nodes:
                            contact_transmissibility += [ self.transmissibility_strengths[0] * self.transmissibility if neighbour in self.no_npi_nodes else self.transmissibility_strengths[1] * self.transmissibility for neighbour in graph.neighbors(i)]
                        else:
                            contact_transmissibility += [ self.transmissibility_strengths[1] * self.transmissibility if neighbour in self.no_npi_nodes else self.transmissibility_strengths[2] * self.transmissibility for neighbour in graph.neighbors(i)]
                    else:
                        neighbours = list(graph.neighbors(i))
                        neighbours_exposed = set(random.sample(neighbours, int(len(neighbours) * self.scale)))
                        contacts += neighbours_exposed
                        if i in self.no_npi_nodes:
                            contact_transmissibility += [ self.transmissibility_strengths[0] * self.transmissibility if neighbour in self.no_npi_nodes else self.transmissibility_strengths[1] * self.transmissibility for neighbour in neighbours_exposed]
                        else:
                            contact_transmissibility += [ self.transmissibility_strengths[1] * self.transmissibility if neighbour in self.no_npi_nodes else self.transmissibility_strengths[2] * self.transmissibility for neighbour in neighbours_exposed]
                    
        self.active_infected = self.active_infected.difference(inactivate)
        contact_probability = np.random.uniform(0, 1, len(contacts))
        success = np.where(contact_probability < contact_transmissibility)
        success_contact = list(np.extract(success, contacts))
        return set(success_contact).intersection(susceptible)


    def run_open_up(self, graph, npis, t_apply_npis, t_open_ups, npi_reopens, **kwargs):
        # FIXME this will not work if the same npi is applied more than once at different strengths
        self.reset(graph)
        npi_idx = 0
        if type(t_apply_npis) == int :
            t_apply_npis = [t_apply_npis]
        t_apply_npi = t_apply_npis[npi_idx]
        npi = npis[npi_idx]

        reopen_idx = 0
        if type(t_open_ups) == int :
            t_open_ups = [t_open_ups]
        t_reopen = t_open_ups[reopen_idx]
        npi_reopen = npi_reopens[reopen_idx]

        for t in range(self.num_days - 1):
            edges_to_remove = []

            while t == t_reopen:
                if npi_reopen == 'hub':
                    graph.add_edges_from(self.removed_edges[npis.index("hub")])
                    self.removed_edges[npis.index("hub")] = []
                elif npi_reopen == "distance":
                    graph.add_edges_from(self.removed_edges[npis.index("distance")])
                    self.removed_edges[npis.index("distance")] = []
                elif npi_reopen == "quarantine":
                    self.quarantine = False
                    graph.add_edges_from(self.removed_edges[npis.index("quarantine")])
                    self.removed_edges[npis.index("quarantine")] = []
                elif npi_reopen == "mask":
                    self.vary_transmissibility = False
                
                reopen_idx += 1
                if reopen_idx >= len(t_open_ups):
                    break
                t_reopen = t_open_ups[reopen_idx]
                npi_reopen = npi_reopens[reopen_idx]

            while t == t_apply_npi:
                if npi == 'hub':
                    hub_edges_to_remove = self.get_edges_to_remove_hub(graph, **kwargs)
                    edges_to_remove += hub_edges_to_remove
                    self.removed_edges.append(hub_edges_to_remove)
                elif npi == "distance":
                    distance_edges_to_remove = self.get_edges_to_remove_social_distancing(graph, **kwargs)
                    edges_to_remove += distance_edges_to_remove
                    self.removed_edges.append(distance_edges_to_remove)
                elif npi == "quarantine":
                    self.quarantine = True
                elif npi == "mask":
                    self.vary_transmissibility = True
                
                npi_idx += 1
                if npi_idx >= len(t_apply_npis):
                    graph.remove_edges_from(edges_to_remove)
                    break
                elif t_apply_npis[npi_idx] != t_apply_npi:
                    graph.remove_edges_from(edges_to_remove)
                t_apply_npi = t_apply_npis[npi_idx]
                npi = npis[npi_idx]            

            if self.quarantine:
                # before = graph.number_of_edges()
                edges_to_remove = self.get_remove_edges_quarantine(graph, **kwargs)
                graph.remove_edges_from(edges_to_remove)
                if len(self.removed_edges) > npis.index("quarantine"):
                    self.removed_edges[npis.index("quarantine")]+= edges_to_remove
                else:
                    self.removed_edges.append(edges_to_remove)
                # after = graph.number_of_edges()
                
            new_exposed = []
            if self.vary_transmissibility:
                new_exposed = self.get_new_exposed_varying_transmissibility(graph, t)
            else:
                new_exposed = self.get_new_exposed(graph, t)
            
            new_infected_s, new_infected_a = self.get_new_infected(t)
            new_recovered_s, new_recovered_a = self.get_new_recovered(t)
            self.update_seir(t, new_exposed, new_infected_s, new_infected_a, new_recovered_s, new_recovered_a)
            self.update_new(new_exposed, new_infected_s, new_infected_a, new_recovered_s, new_recovered_a)
        # return self.to_difference()
        # return self.to_seir()
        return self.to_new(graph.number_of_nodes())

    def run_open_up_with_readjust(self, graph, npis, t_apply_npis, t_open_up, infected, recovered, **kwargs):
        self.reset(graph)
        # print("initial edges", graph.number_of_edges())
        npi_idx = 0
        if type(t_apply_npis) == int :
            t_apply_npis = [t_apply_npis]
        t_apply_npi = t_apply_npis[npi_idx]
        npi = npis[npi_idx]
        for t in range(self.num_days - 1):
            edges_to_remove = []
            if t == t_open_up:
                # print("t_open_up", t_open_up, self.vary_transmissibility, graph.number_of_edges())
                graph.add_edges_from(self.removed_edges)
                # print(graph.number_of_edges())
                self.quarantine = False
                self.vary_transmissibility = False
                self.removed_edges = []
                
                s = self.seir[t][0]
                e = self.seir[t][1]
                i_s = self.seir[t][2]
                i_a = self.seir[t][3]
                r = self.seir[t][4]
                if infected < len(i_s.union(i_a)):
                    nodes_to_move = random.sample(i_s.union(i_a), len(i_s.union(i_a))-infected)
                    i_a = i_a.difference(nodes_to_move)
                    i_s= i_s.difference(nodes_to_move)
                    s = s.union(nodes_to_move)
                elif infected > len(i_s.union(i_a)):
                    nodes_to_move = random.sample(s.union(e), infected-len(i_s.union(i_a)))
                    s = s.difference(nodes_to_move)
                    e = e.difference(nodes_to_move)
                    i_s = i_s.union(nodes_to_move[:int(len(nodes_to_move)/2)])
                    i_a = i_a.union(nodes_to_move[int(len(nodes_to_move)/2):])
                    # self.seir[t-1][2] = self.seir[t-1][2].union(nodes_to_move) 

                if recovered < len(r):
                    nodes_to_move = random.sample(r, len(r)-recovered)
                    r = r.difference(nodes_to_move)
                    s = s.union(nodes_to_move)
                elif recovered > len(r):
                    # print(len(self.seir[t-1][0]), len(self.seir[t-1][1]), len(self.seir[t-1][2]), len(self.seir[t-1][3]))
                    # print(recovered-len(self.seir[t-1][3]))
                    nodes_to_move = random.sample(s.union(e), recovered-len(r))
                    s = s.difference(nodes_to_move[:int(len(nodes_to_move)/2)])
                    e = e.difference(nodes_to_move[int(len(nodes_to_move)/2):])
                    r = r.union(nodes_to_move)
                    # self.seir[t-1][0] = self.seir[t-1][0].difference(nodes_to_move)
                    # self.seir[t-1][1] = self.seir[t-1][1].difference(nodes_to_move)
                    # self.seir[t-1][2] = self.seir[t-1][2].difference(nodes_to_move)
                    # self.seir[t-1][3] = self.seir[t-1][3].union(nodes_to_move)
                self.seir[t][0] = s
                self.seir[t][1] = e
                self.seir[t][2] = i_s
                self.seir[t][3] = i_a
                self.seir[t][4] = r
                

            while t == t_apply_npi:
                if npi == 'hub':
                    edges_to_remove += self.get_edges_to_remove_hub(graph, **kwargs)
                elif npi == "distance":
                    edges_to_remove += self.get_edges_to_remove_social_distancing(graph, **kwargs)
                elif npi == "quarantine":
                    self.quarantine = True
                elif npi == "mask":
                    self.vary_transmissibility = True
                
                npi_idx += 1
                if npi_idx >= len(t_apply_npis):
                    graph.remove_edges_from(edges_to_remove)
                    self.removed_edges = self.removed_edges + edges_to_remove
                    break
                t_apply_npi = t_apply_npis[npi_idx]
                npi = npis[npi_idx]            

            if self.quarantine:
                edges_to_remove = self.get_remove_edges_quarantine(graph, **kwargs)
                graph.remove_edges_from(edges_to_remove)
                self.removed_edges = self.removed_edges + edges_to_remove
                
            new_exposed = []
            if self.vary_transmissibility:
                new_exposed = self.get_new_exposed_varying_transmissibility(graph, t)
            else:
                new_exposed = self.get_new_exposed(graph, t)
            new_infected_s, new_infected_a = self.get_new_infected(t)
            new_recovered_s, new_recovered_a = self.get_new_recovered(t)
            self.update_seir(t, new_exposed, new_infected_s, new_infected_a, new_recovered_s, new_recovered_a)
            self.update_new_infected(new_infected_s, new_infected_a)
        # print("self.vary_transmissibility at end", self.vary_transmissibility)
        # print("graph.number_of_edges() at end ", graph.number_of_edges())
        # return self.to_seir()

    # def run_dynamic_reopening(self, graph, npis, t_apply_npis, t_open_up, threshold=770/17800, **kwargs):
    #     self.reset(graph)
    #     npi_idx = 0
    #     if type(t_apply_npis) == int :
    #         t_apply_npis = [t_apply_npis]
    #     t_apply_npi = t_apply_npis[npi_idx]
    #     npi = npis[npi_idx]
    #     for t in range(self.num_days - 1):
    #         edges_to_remove = []
    #         if t == t_open_up:
    #             graph.add_edges_from(self.removed_edges)
    #             self.quarantine = False
    #             self.vary_transmissibility = False
    #             self.reopened = True

    #         while t == t_apply_npi:
    #             if npi == 'hub':
    #                 edges_to_remove += self.get_edges_to_remove_hub(graph, **kwargs)
    #             elif npi == "distance":
    #                 edges_to_remove += self.get_edges_to_remove_social_distancing(graph, **kwargs)
    #             elif npi == "quarantine":
    #                 self.quarantine = True
    #             elif npi == "mask":
    #                 self.vary_transmissibility = True
                
    #             npi_idx += 1
    #             if npi_idx >= len(t_apply_npis):
    #                 graph.remove_edges_from(edges_to_remove)
    #                 self.removed_edges = self.removed_edges + edges_to_remove
    #                 break
    #             t_apply_npi = t_apply_npis[npi_idx]
    #             npi = npis[npi_idx]            

    #         if self.quarantine:
    #             edges_to_remove = self.get_remove_edges_quarantine(graph, **kwargs)
    #             graph.remove_edges_from(edges_to_remove)
    #             self.removed_edges = self.removed_edges + edges_to_remove
                
    #         new_exposed = []
    #         if self.vary_transmissibility:
    #             new_exposed = self.get_new_exposed_varying_transmissibility(graph, t)
    #         else:
    #             new_exposed = self.get_new_exposed(graph, t)
    #         new_infected_s, new_infected_a = self.get_new_infected(t)
    #         new_recovered_s, new_recovered_a = self.get_new_recovered(t)
    #         self.update_seir(t, new_exposed, new_infected_s, new_infected_a, new_recovered_s, new_recovered_a)

    #         if self.reopened == True and (self.seir[t][2] + self.seir[t][3])/graph.number_of_nodes() >= threshold:
    #             graph.remove_edges_from(self.removed_edges)
    #             self.reopened == False
    #         elif t > t_open_up and self.reopened == False and (self.seir[t][2] + self.seir[t][3])/graph.number_of_nodes() < threshold:
    #             graph.add_edges_from(self.removed_edges)
    #             if "quarantine" in npis:
    #                 self.quarantine = True
    #             if "mask" in npis:
    #                 self.vary_transmissibility = True
    #             self.reopened == True
    #     return self.to_seir()



    def get_remove_edges_quarantine(self, graph, **kwargs):
        percent_remove = kwargs['quarantine_p']
        exposed = self.seir[-1][1]
        infected = self.seir[-1][2] | self.seir[-1][3]
        num_remove_exposed = int(np.rint(len(exposed) * percent_remove))
        num_remove_infected = int(np.rint(len(infected) * percent_remove))
        exposed_nodes_to_remove_edge = random.sample(exposed, num_remove_exposed)
        infected_nodes_to_remove_edge = random.sample(infected, num_remove_infected)
        if len(self.no_npi_nodes) > 0:
            exposed_nodes_to_remove_edge = set(exposed_nodes_to_remove_edge) - self.no_npi_nodes
            infected_nodes_to_remove_edge = set(infected_nodes_to_remove_edge) - self.no_npi_nodes
        edges_to_remove = list(graph.edges(exposed_nodes_to_remove_edge)) + list(graph.edges(infected_nodes_to_remove_edge))
        return edges_to_remove


    def to_seiir(self):
        seir_0 = np.zeros((self.num_days, len(SEIR_COMPARTMENTS)+1))
        for t in range(self.num_days):
            s, e, i_s, i_a, r = len(self.seir[t][0]), \
                         len(self.seir[t][1]), \
                         len(self.seir[t][2]), \
                         len(self.seir[t][3]), \
                         len(self.seir[t][4])
            seir_0[t] = [s, e, i_s, i_a, r]
        return seir_0 / np.expand_dims(np.sum(seir_0, axis=1), 1)

    def to_new(self, n):
        return np.array(self.new_seir) / n