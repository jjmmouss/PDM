import networkx as nx
import re
import itertools
import math

'''
Load the cascade files and set it up for the NetRate variation algorithms (SGD,CVX,ADMM)

Input : 
    -file : path to the .txt file containing the record of the cascades.
Output :
    -G : Directed graph containing only the vertices that were used during the cascades.
    -C_dic : A dictionary index on the cascade number (first cascade, seconde,.) containing dictionary of the form {vertex: time of infection}

'''
def Init(file) :
    G,cascades_list = load_cascade_from_file(file)
    #Generates the DAG and Trees for each cascade and stores it into a dic. (DAG,Tree)
    C_dic = {}
    
    for index,c in enumerate(cascades_list):
        C = Create_dic_from_cascade(c[0]) 
        C_dic[index] = C
    return G,C_dic


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
def Create_dic_from_cascade(cascade):
    cascade = cascade.split(";")
    data_cascade = {}
    for elements in cascade:
        elements = elements.split(',')
        data_cascade[int(elements[0])] = float(elements[1]) #(vertex,time)
    return data_cascade

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
            alpha_if = edge[2][1:-1]
        except :
            print("No edge transmission parameter")
            alpha_if = 0
        G.add_edge(int(vertex_i),int(vertex_f),weight =float(alpha_if))
    return G