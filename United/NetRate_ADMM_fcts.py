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
import random


'''
Create the matrix Mi for the ADMM method

Each cell k,l in Mi is one if in cascade k node l infected node i, 0 otherwise

Input :
    cascade_dic : a dictionnary of all cascade (in a graph object, DAG)
    node : an int describing which node we are considering in the ADMM iteration
    number_of_nodes : number of nodes in the underlying network (we assume that rhe union of all cascades cover all the nodes)
'''
def Create_matrix_Mi_and_Ti(cascade_dic,node,number_of_nodes,window) : 
    M_i = np.zeros((len(cascade_dic),number_of_nodes))
    T_i = np.zeros((len(cascade_dic),number_of_nodes))
    index_to_delete =[]
    for cascade in cascade_dic :
        c = cascade_dic[cascade] # graph object
        if node in c.nodes():
            t_i = c.node[node]["time"]
            parent_list = list(c.predecessors(node)) # create a list of all nodes (int) that where infected before node i in the cascade
            for parent in parent_list :
                t_parent = c.node[parent]["time"]
                M_i[cascade,parent] = 1
                T_i[cascade,parent] = (t_i-t_parent)
                if (t_i-t_parent)<=0:
                    print("Time error, the flow of time is reversed the world's end is near")
        else :
            for j in c.nodes() :
                t_j = c.node[j]["time"]
                T_i[cascade,j] = (window-t_j) # SIGNE is now correct !
                
        if np.all(M_i[cascade,:]==0):
            index_to_delete.append(cascade)
    M_i = np.delete(M_i,(index_to_delete),axis=0)
                
    return M_i,T_i

def Compute_objective_value_for_one_node(alpha_mat_i,cascade_dic,window,eps,node_i) :
    obj_i = 0
    for cascs in cascade_dic :
        c = cascade_dic[cascs]
        if node_i in c.nodes():
            t_i = c.nodes[node_i]["time"]
            parents_i_list = list(c.predecessors(node_i))
            sum_tmp = 0
            for parent in parents_i_list :
                t_parent = c.nodes[parent]["time"]
                obj_i += alpha_mat_i[parent]*(t_i-t_parent)
                sum_tmp += alpha_mat_i[parent]
            if sum_tmp<=0 :
                if len(parents_i_list)>0 :
#                     print("Attention log ill defined incoming")
                    obj_i -=math.log(eps)
            else :
                obj_i -= math.log(sum_tmp)
        else :
            for j in c.nodes():
                t_j = c.nodes[j]["time"]
                obj_i += alpha_mat_i[j]*(window-t_j)
    return obj_i

def Compute_true_objective_value_per_node(G_true,cascade_dic,window,node_i):
    obj_i = 0
    for cascs in cascade_dic :
        c = cascade_dic[cascs]
        if node_i in c.nodes():
            t_i = c.nodes[node_i]["time"]
            parent_i_list = list(c.predecessors(node_i))
            sum_tmp = 0
            for parent in parent_i_list :
                if (parent,node_i) in G_true.edges():
                    t_parent = c.nodes[parent]["time"]
                    try :
                        alpha_ji = G_true.edges[(parent,node_i)]["weight"][0]
                    except TypeError :
                        alpha_ji = G_true.edges[(parent,node_i)]["weight"]
                    obj_i += alpha_ji*(t_i-t_parent)
                    sum_tmp += alpha_ji
            if sum_tmp == 0:
                if len(parent_i_list)>0 :
                    print("Oulala")
            else :
                obj_i -= math.log(sum_tmp)
        else :
            for j in c.nodes() :
                if (j,i) in G_true.edges():
                    t_j = c.nodes[j]["time"]
                    try :
                        alpha_ji = G_true.edges[(j,i)]["weight"][0]
                    except TypeError :
                        alpha_ji = G_true.edges[(j,i)]["weight"]
                    obj_i += alpha_ji*(window-t_j)
    return obj_i
def Compute_recall_precision(G_true,A_hat,tol):
    correct = 0
    nb_edges = 0
    for i in range(0,G_true.number_of_nodes()):
        for j in range(0,G_true.number_of_nodes()):
            if A_hat[i,j] > tol:
                nb_edges +=1
                if (i,j) in G_true.edges():
                    correct +=1
    precision = correct/nb_edges
    recall = correct/G_true.number_of_edges()
    return(recall,precision)
    
def NetRate_ADMM(G_true,N,alpha_init,DAG_C_dic,window,iter_GD,iter_ADMM,gamma,alpha_max,u,eps,tol):
    dic_of_obj_per_node_per_iter = {}
    recall_precision_list = []
    obj_per_node = []
    A_hat = np.zeros((N,N))
    for i in G_true.nodes :
        #print("Node : ",i)
        #t_start_node_i = time.time()
        dic_of_obj_per_node_per_iter[i] = []


        '''
        initialization
        '''
        a_k = np.ones((N,1))*alpha_init
        a_k[i]=0
        M_i,T_i = Create_matrix_Mi_and_Ti(DAG_C_dic,i,N,window) # TO do : consider to use sparse matrix  
        grad_i = (np.sum(T_i,axis=0).T)
        for bla in range(0,len(grad_i)):
            if grad_i[bla]==0 :
                a_k[bla]=0

        z = a_k.copy()         
        rho = np.zeros((M_i.shape[0],1))
        S_i = np.matmul(M_i,z)

        has_converge=False
        k=0
        #Start iteration of ADMM
        while has_converge==False and k<iter_ADMM :
            '''
            Update alpha using gradient descent
            '''
            grad = grad_i + np.matmul(rho.T,M_i)
            for j in range(0,iter_GD):
                grad_j = grad - u*(np.matmul(M_i.T,(S_i-np.matmul(M_i,a_k))).T) # sign is correct
                a_k = a_k - gamma*grad_j.T
                a_k = np.maximum(a_k,0)
                a_k = np.minimum(a_k,alpha_max)

            '''
            update S_i and rho for each cascades via the closed form formula and the gradient descent respectively
            '''
            for cascs in range(0,M_i.shape[0]) :
                c = DAG_C_dic[cascs]
                Malpha = np.matmul(M_i[cascs,:],a_k)
    #             Malpha = 0
    #             if i in c.nodes :
    #                 parent_i_c = list(c.predecessors(i))
    #                 for papa in parent_i_c :
    #                     Malpha += a_k[papa]
    #             sqrt_delta = math.sqrt((rho[cascs]+Malpha)**2 + 4*u)
    #             S_i[cascs] = ((rho[cascs]+Malpha)+sqrt_delta)/(2*u)
    #             if i in c.nodes() and len(list(c.predecessors(i)))>0 :
                sqrt_delta = math.sqrt((rho[cascs]+u*Malpha)**2+4*u)
                S_i[cascs] = (rho[cascs]+u*Malpha + sqrt_delta)/(2*u)
                if S_i[cascs]<0 :
                    print("Huston Huston we have a probleme")
                if S_i[cascs] ==0 :
                    print("one component of S_i is zero")
    #                 rho[cascs] = rho[cascs]-u*(S_i[cascs]-Malpha)
    #             else :
    #                 S_i[cascs] = (rho[cascs]+u*Malpha)/u

                rho[cascs] = rho[cascs]-u*(S_i[cascs]-Malpha)

            '''compute the objective function for node i in iteration k'''
            if k%5 ==0 :
                #print(k)
                obj_i_k = Compute_objective_value_for_one_node(a_k,DAG_C_dic,window,eps,i)
                dic_of_obj_per_node_per_iter[i].append(obj_i_k)
            k+=1
            if len(dic_of_obj_per_node_per_iter[i])>=2:
                delta = abs(dic_of_obj_per_node_per_iter[i][-1]-dic_of_obj_per_node_per_iter[i][-2])
                if delta<1e-3 :
                    has_converge=True
        A_hat[:,i] = a_k.flatten()
#         (rec,prec) = Compute_recall_precision(G_true,A_hat,tol)
#         recall_precision_list.append((rec,prec))
        #t_end_node_i = time.time()
        #print("computation time for node i : ", t_end_node_i-t_start_node_i)
    return A_hat,dic_of_obj_per_node_per_iter,recall_precision_list