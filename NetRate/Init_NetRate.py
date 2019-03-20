import networkx as nx
import re
import itertools
import math

def Init(file) :
    G,cascades_list = load_cascade_from_file(file)
    #Generates the DAG and Trees for each cascade and stores it into a dic. (DAG,Tree)
    DAG_c_dic = {}
    
    for index,c in enumerate(cascades_list):
        DAG_c = Create_DAG_from_cascade(c[0]) 
        DAG_c_dic[index] = DAG_c
    return G,DAG_c_dic

def load_cascade_from_file(file): #pseudo code version, needs to be updated
    f = open(file,"r")
    cascade_list = []
    G = nx.DiGraph()
    #Reads the file and set up the nodes as well as the format for the cascades
    for nodes in f:
        if not nodes.strip(): ## Stop at the first blank line
            print("All nodes were read")
            break
        node = re.split(',|\n',nodes) # the format of the input file is <id>,<name>  
        vertex = node[0]
        names = node[1]
        G.add_node(int(vertex),name = names)
    for cascades in f:
        cascades = cascades.split("\n")
        cascade_list.append(cascades[:-1])# small adjustment to make a standard format (i.e remove \n at the end)
    return G,cascade_list

'''

'''
def Create_DAG_from_cascade(cascade):
    DAG = nx.DiGraph()
    cascade = cascade.split(";")
    data_cascade = []
    for elements in cascade:
        elements = elements.split(',')
        data_cascade.append([int(elements[0]),float(elements[1])])
        DAG.add_node(int(elements[0]),time = float(elements[1]))
    for couple1 in data_cascade :
        vertex1,time1 = couple1
        for couple2 in data_cascade :
            vertex2,time2 = couple2
            if time1<time2 : #The propagation can only move forward
                DAG.add_edge(vertex1,vertex2)
    return DAG

def Load_ground_truth(file):
    f = open(file,"r")
    G = nx.DiGraph()
    for nodes in f:
        if not nodes.strip(): ## Stop at the first blank line
            break
        node = re.split(',|\n',nodes) # the format of the input file is <id>,<name>  
        vertex = node[0]
        names = node[1]
        G.add_node(int(vertex),name = names)
    for edges in f :
        edge = re.split(',|\n',edges)
        vertex_i = edge[0] # initial vertex of the directed edge
        vertex_f = edge[1] # final vertex of the edge
        try :
            alpha_if = edge[2]
        except :
            print("No edge transmission parameter")
            alpha_if = 0
        G.add_edge(int(vertex_i),int(vertex_f),weight =alpha_if)
    return G