import networkx as nx # version 2.2
import matplotlib.pyplot as plt
import re
import cvxpy as cp
import operator #to sort elements in a list of tuples
import itertools
import math
import numpy as np
import os
import sys
import time

'''
Evaluate the objective function given a cascade and the alphas

Input :
    -alpha_mat : matrix of size |V|*|V| containing all the alpha_i,j
    -cascade_graph : Directed graph object (DAG), of a cascade
    -G_star : Directed graph containing only the nodes
    -window : int representing the time limit until which we observe a cascade propagation
Output :
    -obj : value of the objective funtion restricted to one cascade
'''
def Objective_function(alpha_mat,cascade_graph_c,G_true,window,eps):
    sum_psi1 = 0
    sum_psi2 = 0
    sum_psi3 = 0
    for i in cascade_graph_c.nodes():
        sum_psi3_tmp = 0
        t_i = cascade_graph_c.nodes[i]["time"]
        for m in G_true.nodes():
            if alpha_mat[i,m]==eps :
                alpha_mat[i,m] = 0
            if alpha_mat[m,i]==eps:
                alpha_mat[m,i] = 0
            if m not in cascade_graph_c.nodes():
                sum_psi1 += -alpha_mat[i,m]*(window-t_i)
            elif (m,i) in cascade_graph_c.edges():
                t_m = cascade_graph_c.nodes[m]["time"]
                sum_psi2 += -alpha_mat[m,i]*(t_i-t_m)
                if alpha_mat[m,i]==0:
                    sum_psi3_tmp+=eps
                else:
                    sum_psi3_tmp += alpha_mat[m,i]
        if sum_psi3_tmp!=0:
            sum_psi3 += math.log(sum_psi3_tmp)
        else :
            if len(list(cascade_graph_c.predecessors(i)))>0:
                print("log of zero")
                sum_psi3 = -1e10 # should mimic the - infinity
    obj = sum_psi1+sum_psi2+sum_psi3
    return obj
def True_objective_value(G_true,cascade_graph_dic,window,eps):
    A_true = np.zeros((G_true.number_of_nodes(),G_true.number_of_nodes()))
    for edge in G_true.edges():
        try :
            A_true[edge[0],edge[1]] = G_true.edges[edge[0],edge[1]]["weight"][0]
        
        except TypeError:
            A_true[edge[0],edge[1]] = G_true.edges[edge[0],edge[1]]["weight"]
            
    true_obj = 0
    for c in cascade_graph_dic:
        cascade = cascade_graph_dic[c]
        true_obj += Objective_function(A_true,cascade,G_true,window,eps)
    return true_obj
def Infered_objective_value(alpha_mat,G_true,cascade_graph_dic,window,eps):
    obj = 0
    for c in cascade_graph_dic:
        cascade = cascade_graph_dic[c]
        obj += Objective_function(alpha_mat,cascade,G_true,window,eps)
    return obj

def Netrate_SG(DAG_C_dic,K,G_true,alpha_init,gamma,eps,window,N,max_alpha):
    k=0
    c_index_list = np.random.choice(list(range(0,len(DAG_C_dic))),K,replace = True)
    obj_per_itter = []
    obj_fct = 0
    A_hat = np.zeros((N,N))
    while k <K :
        c_index = c_index_list[k]
        DAG_c = DAG_C_dic[c_index]
        for i in G_true.nodes():
            if i in DAG_c.nodes(): # case of infected node
                t_i = DAG_c.nodes[i]["time"]
                parents = list(DAG_c.predecessors(i))
                sum_grad = 0
                for papa in parents :
                    if A_hat[papa,i]==0 :
                        A_hat[papa,i] = alpha_init
                    sum_grad += A_hat[papa,i]
                for papa in parents :
                    t_papa = DAG_c.nodes[papa]["time"]
                    if t_i-t_papa<=0:
                        print("Time Error ")
                        print("t_i is :", t_i)
                        print("parent infection time is : ",t_papa)
                    A_hat[papa,i] = max(A_hat[papa,i]-gamma*((t_i-t_papa)-1/sum_grad),eps) #SG with grad of infected nodes
                    if A_hat[papa,i]> max_alpha:
                        A_hat[papa,i] = max_alpha
            else : # case of uninfected nodes
                for j in DAG_c.nodes():
                    t_j = DAG_c.nodes[j]["time"]
                    if window-t_j <0 :
                        print ("Time Error")
                        print("window is :",window)
                        print("time of infection of node j is : ",t_j)
                    A_hat[j,i] = max(A_hat[j,i]-gamma*(window-t_j),eps) # SG with grad for !infected nodes (can only decrease)

#         if k%100==0:
#             print("itteration ", k)
#         if k%500 ==0:
#             obj_fct = Infered_objective_value(A_hat,G_true,DAG_C_dic,window,eps)
#             obj_per_itter.append(obj_fct)
#             print("objective function is : ",obj_fct) # Remeber, we try to maximize this value
        k+=1
    return A_hat