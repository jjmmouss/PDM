import networkx as nx
import matplotlib.pyplot as plt
import re
import random
import operator #to sort elements in a list of tuples
import itertools
import math
import numpy as np
'''
Generates one cascade

Input :
    -ground_truth : graph of the true underlying network
    -beta : probability of an edge propagating the infection
    -alpha : parameter of the exponential (and power) law
    -window : arbitrally large value after which the cascade stops
    -model : for now take only value 0
Output :
    -cascade_graph : The graph of the cascade where each nodes store their time of infection
'''
def Generate_one_cascade(ground_truth,beta,alpha,window,model) :
#     ground_truth = G_true
#     beta = 0.5 #here beta is assumed to be constant for all edges
#     alpha = 1.0 #Here alpha is assumed to be constant for all edges
#     window = 100
#     model = 0
    list_of_infected_nodes = []
    cascade_graph = nx.DiGraph()
    cascade_graph.clear()
    if ground_truth.number_of_nodes()==0 :
        print("Error")
        return
    while(len(list_of_infected_nodes)<2) :
        list_of_infected_nodes = []
        cascade_graph.clear()
        global_time = 0
        Start_node = int(random.choice(ground_truth.nodes)["name"])
        list_of_infected_nodes.append((Start_node,global_time))
        cascade_graph.add_node(Start_node,time = global_time) #add the infection node to the cascade graph
        while(True) :
            list_of_infected_nodes.sort(key = operator.itemgetter(1)) #Sort the list of infected node based on the time of infection
            node_infected,time_of_infection = list_of_infected_nodes[0] #get the oldest node that has not propagate the infection yet
            if time_of_infection>=window : #Time limite of the infection propagation
                break
            sucessors = list(ground_truth.successors(node_infected))
            for neighbours in sucessors :
                rand_chance = np.random.uniform(0,1,1)
                if(rand_chance>=beta): #Check if the edge propagates the diseases (flip of coin)
                    continue
                sigmaT = 0
                if model ==0 : #Exponential case
                        sigmaT = np.random.exponential(alpha) #TO DO : check if alpha or 1/alpha
                T1 = time_of_infection + sigmaT
                if neighbours in cascade_graph.nodes():
                    time_of_previous_infection = cascade_graph.nodes[neighbours]["time"]
                    if T1 >= time_of_previous_infection :
                        continue
                    else :
                        parent = list(cascade_graph.predecessors(neighbours))[0]
                        cascade_graph.remove_edge(parent,neighbours)
                        cascade_graph.remove_node(neighbours)
                cascade_graph.add_node(neighbours,time = T1)
                cascade_graph.add_edge(node_infected,neighbours)
                list_of_infected_nodes.append((neighbours,T1))
            list_of_infected_nodes[0] = (node_infected,window) #Place the node that we handled at the end of the queue     
    return cascade_graph


'''
Generale a set of cascades until a ratio of edge of the true network has been used at least once.

Input :
    -ground_thruth : True underlying network, directed graph
    -ratio_of_used_edges : int giving the number (0% to 100%) of edges we want to be used at least once
    -beta, alpha, window : int that describes the model. 
                            Beta is the proba of an edge transmitting the infection 
                            Alpha is the parameter of the exp law or power law
                            window is an arbitrary large number after which the cascades stop (used to keep track of all infected nodes without having to delete them)
    -model : for now model take only the value 0 (for the exponential law). Could later take value 1 for the power law
    
Output :
    -dic_of_cascades : dictionnary of all cascades. One entery contains a list of tuples (vertex,time of infection)
'''
def Generate_all_cascades(ground_truth,ratio_of_used_edges,beta,alpha,window,model):
    nb_used_edges = 0
    nb_itt = 0
    dic_of_cascades = {}
    list_of_used_edges = []
    while (nb_used_edges<ratio_of_used_edges/100*ground_truth.number_of_edges()):
        cascade = Generate_one_cascade(ground_truth,beta,alpha,window,model)
        list_of_tuples = []
        for node in cascade.nodes():
            vertex = node
            t_v = cascade.nodes[node]["time"]
            list_of_tuples.append((vertex,t_v))
        dic_of_cascades[nb_itt] = list_of_tuples
        for edge in cascade.edges():
            if edge not in list_of_used_edges :
                list_of_used_edges.append(edge)
        nb_used_edges = len(list_of_used_edges)
        nb_itt+=1
    return dic_of_cascades

'''
Once a list of cascade was generate, saves it (in the right format) to a file

Input : 
    -file_name : the name of the text file we are going to save the cascades in
    -dic_of_cascades : a dictionary of all cascades generated. An entery of the dic is a list of tuples (vertex,time of infection)
    -ground_truth : The true underlying network (only used to get the infos of all nodes)
'''
def Save_cascade_to_file(file_name,dic_of_cascades,ground_truth):
    f = open(file_name,"w")
    for nodes in ground_truth.nodes() :
        f.write(str(nodes)+","+str(nodes)+"\n")
    f.write("\n")
    for keys in dic_of_cascades :
        c = dic_of_cascades[keys]
        for couple in c :
            v,t_v = couple
            if couple == c[-1] :
                f.write(str(v)+","+str(t_v))
            else :
                f.write(str(v)+","+str(t_v)+";")
        f.write("\n")
    f.close()
    
###################### Graph generation (ground truth) #######################

def Generate_random_graph(nb_vertex,nb_edges):
    list_vertex = list(range(nb_vertex))
    max_number_edge = len(list(itertools.product(list_vertex,list_vertex)))-len(list_vertex)
    if nb_edges>max_number_edge :
        print("You ask for more edge than it is possible")
        return 0
    G = nx.DiGraph()
    for i in range(0,nb_vertex):
        G.add_node(i,name = str(i))
    while len(G.edges)<nb_edges :
        v1 = int(np.random.choice(G.nodes))
        v2 = int(np.random.choice(G.nodes))
        while (v1,v2) in G.edges or v1==v2:
            v2 = int(np.random.choice(G.nodes))
        G.add_edge(v1,v2)
    return G

def Save_graph_to_file(file_name,G) :
    f = open(file_name,"w")
    for nodes in G.nodes():
        f.write(str(nodes)+","+str(nodes)+"\n")
    f.write("\n") #end of the node so we add an empty line
    for edge in G.edges():
        f.write(str(edge[0])+","+str(edge[1])+"\n")
    f.close()