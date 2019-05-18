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
def Objective_function(alpha_mat,cascade_c,N,window,eps):
    sum_psi1 = 0
    sum_psi2 = 0
    sum_psi3 = 0
    has_parent = False
    for i in cascade_c.keys():
        sum_psi3_tmp = 0
        t_i = cascade_c[i]
        for m in range(0,N):
            if alpha_mat[i,m]==eps :
                alpha_mat[i,m] = 0
            if alpha_mat[m,i]==eps:
                alpha_mat[m,i] = 0
            if m not in cascade_c.keys():
                sum_psi1 += -alpha_mat[i,m]*(window-t_i)
            elif m in cascade_c.keys() and cascade_c[m]<t_i:
                has_parent = True
                t_m = cascade_c[m]
                sum_psi2 += -alpha_mat[m,i]*(t_i-t_m)
                if alpha_mat[m,i]==0:
                    sum_psi3_tmp+=eps
                else:
                    sum_psi3_tmp += alpha_mat[m,i]
        if sum_psi3_tmp!=0:
            sum_psi3 += math.log(sum_psi3_tmp)
        else :
            if has_parent :
                print("log of zero")
                sum_psi3 = -1e10 # should mimic the - infinity
    obj = sum_psi1+sum_psi2+sum_psi3
    return obj

def True_objective_value(G_true,cascade_dic,window,eps):
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

def Infered_objective_value(alpha_mat,N,cascade_dic,window,eps):
    obj = 0
    for c in cascade_dic:
        cascade = cascade_graph_dic[c]
        obj += Objective_function(alpha_mat,cascade,N,window,eps)
    return obj


def Compute_Acc(G_true,A_hat,tol) :
    correct = 0
    nb_edges = 0
    top_acc = 0
    bot_acc = 0
    for i in range(0,G_true.number_of_nodes()):
        for j in range(0,G_true.number_of_nodes()):
            if A_hat[i,j] >tol:
                nb_edges +=1
                bot_acc +=1
                if (i,j) in G_true.edges():
                    correct +=1
                    bot_acc+=1
                else :
                    top_acc+=1
            else :
                if (i,j) in G_true.edges():
                    top_acc +=1
                    bot_acc+=1
                
    try :
        precision = correct/nb_edges
    except :
        precision = 0
    recall = correct/G_true.number_of_edges()
    accuracy = 1-(top_acc/bot_acc)
    mse = 0
    nb_correcte_edges=0
    for edge in G_true.edges():
        try:
            true_alpha = G_true.edges[edge]["weight"][0]
        except TypeError:
            true_alpha = G_true.edges[edge]["weight"]
        approx_alpha = A_hat[edge[0],edge[1]]
        mse += pow((true_alpha-approx_alpha),2) # mean square error
    try :
        mse = mse/G_true.number_of_edges()
    except ZeroDivisionError:
        print ("There is no correct edge")
    return (precision,recall,accuracy,mse)



def Netrate_SGD(G_true,C_dic,N,SGD_param):
    alpha_init,max_alpha,K,gamma,eps,window,tol,more_infos = SGD_param
    k=0
    c_index_list = np.random.choice(list(range(0,len(C_dic))),K,replace = True)
    obj_per_itter = []
    obj_fct = 0
    Acc_MSE_time = []
    A_hat = np.zeros((N,N))
    
    while k <K :
        t_s = time.time()
        c_index = c_index_list[k]
        cascade = C_dic[c_index]
        for i in G_true.nodes():
            if i in cascade.keys(): # case of infected node
                t_i = cascade[i]
                parent_i = []
                sum_grad = 0
                for infected in cascade :
                    t_j = cascade[infected]
                    if t_j<t_i : #potential parent
                        parent_i.append(infected)
                        if A_hat[infected,i]==0 :
                            A_hat[infected,i] = alpha_init
                        sum_grad += A_hat[infected,i]
                for papa in parent_i :
                    t_papa = cascade[papa]
                    if t_i-t_papa<=0:
                        print("Time Error ")
                        print("t_i is :", t_i)
                        print("parent infection time is : ",t_papa)
                    A_hat[papa,i] = max(A_hat[papa,i]-gamma*((t_i-t_papa)-1/sum_grad),eps) #SG with grad of infected nodes
                    if A_hat[papa,i]> max_alpha:
                        A_hat[papa,i] = max_alpha
            else : # case of uninfected nodes
                for j in cascade.keys():
                    t_j = cascade[j]
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
#             print("objective function is : ",obj_fct) 
        k+=1
        t_f = time.time()
        time_iter = t_f-t_s
        if more_infos :
            (precision,recall,accuracy,mse) = Compute_Acc(G_true,A_hat,tol)
            Acc_MSE_time.append((accuracy,mse,time_iter))
    return A_hat,Acc_MSE_time