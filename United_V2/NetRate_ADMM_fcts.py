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
Create the matrices Mi and T_i for the ADMM method

Each cell k,l in Mi is one if in cascade k node l infected node i, 0 otherwise.
Each cell k,l in Ti is either (t_i-t_l) if node l was infected before node i in cascade k. or (T-t_l) if node i was not infected during the cascade.

T_i can be seen as an fusion of psi 1 and psi 2 in the paper (NetRate). M_i is here to do the sumation in psi3

Input :
    cascade_dic : a dictionnary of all cascade (in a graph object, DAG)
    node : an int describing which node we are considering in the ADMM iteration
    number_of_nodes : number of nodes in the underlying network (we assume that rhe union of all cascades cover all the nodes)
output :
    M_i,T_i : matrices of size N X N for the computation of ADMM.
'''
def Create_matrix_Mi_and_Ti(cascade_dic,node,number_of_nodes,window) : 
    M_i = np.zeros((len(cascade_dic),number_of_nodes))
    T_i = np.zeros((len(cascade_dic),number_of_nodes))
    index_to_delete =[]
    for cascade in cascade_dic :
        c = cascade_dic[cascade] # dic object of the form {vertex: time}
        if node in c.keys():
            t_i = c[node]
            for infected in c :
                t_j = c[infected]
                if t_j<t_i :
                    M_i[cascade,infected] = 1
                    T_i[cascade,infected] = (t_i-t_j)
                    if (t_i-t_j)<=0:
                        print("Time error, the flow of time is reversed the world's end is near")
        else :
            for j in c :
                t_j = c[j]
                T_i[cascade,j] = (window-t_j) # SIGNE is now correct !
                
        if np.all(M_i[cascade,:]==0):
            index_to_delete.append(cascade) # reduce the size of the problem if a node was never a potential parent of i. 
    M_i = np.delete(M_i,(index_to_delete),axis=0)
                
    return M_i,T_i

'''
Given the true underlying network and the current estimation of the variables alpha computes the precision, recall, accuracy and mean square error.

Input :
    -G_true : Graph object corresponding to the underlying network
    -A_hat : matrix of size N X N which's entery (i,j) is the values of alpha_ij

Output:
    -precision,recall,accuracy,mse : values of the precision, recall, accuracy and mean square error
'''
def Compute_precision_recall_acc_mse(G_true,A_hat,tol):
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
    for edge in G_true.edges():
            try:
                true_alpha = G_true.edges[edge]["weight"][0]
            except TypeError:
                true_alpha = G_true.edges[edge]["weight"]
            approx_alpha = A_hat[edge[0],edge[1]]
            mse += pow((true_alpha-approx_alpha),2) #  square error
    try :
        mse = mse/G_true.number_of_edges() #mean square error
    except ZeroDivisionError:
        print ("There is no correct edge")
    return (precision,recall,accuracy,mse)


def NetRate_ADMM(G_true,DAG_C_dic,N,param_ADMM,Lagrangian):
        
    alpha_init,alpha_max,window,iter_GD,iter_ADMM,gamma,u,eps,tol,more_infos,line_search = param_ADMM
    
    dic_of_obj_per_node_per_iter = {}
    lag_per_node_per_iter = {}
    recall_precision_list = []
    obj_per_node = []
    acc_MSE_time = []
  
    A_hat = np.zeros((N,N)) #A_hat will store all the value of the differents alphas.
    for i in range(0,N) :
        t_s = time.time()

        dic_of_obj_per_node_per_iter[i] = []
        lag_per_node_per_iter[i] = []

        '''
        initialization
        '''
        a_k = np.ones((N,1))*alpha_init
        a_k[i]=0 # there is no self looping edge
        
        M_i,T_i = Create_matrix_Mi_and_Ti(DAG_C_dic,i,N,window) # TO do : consider to use sparse matrix  
        grad_i = (np.sum(T_i,axis=0).T) # sum of (T-t_j) and (t_i-t_j) i.e grad of psi1 + psi2
        
        for bla in range(0,len(grad_i)): #Should not happended (I think). Safety measure.
            if grad_i[bla]==0 :
                a_k[bla]=0

        S_i = np.matmul(M_i,a_k) # Second variable of the ADMM optimization problem.
        rho = np.zeros((M_i.shape[0],1)) # Third variable of the ADMM optimization problem.
        
        has_converge=False
        k=0
        obj_i_initial_value = Compute_objective_value_for_one_node(a_k,DAG_C_dic,window,eps,i)
        dic_of_obj_per_node_per_iter[i].append(obj_i_initial_value)
        #Start iteration of ADMM
        while has_converge==False and k<iter_ADMM :
            '''
            Update alpha using gradient descent
            '''
            grad = grad_i + np.matmul(rho.T,M_i)
            for j in range(0,iter_GD):
                grad_j = grad - u*(np.matmul(M_i.T,(S_i-np.matmul(M_i,a_k))).T) # sign is correct
                if j==0 and k==0 and line_search == True : #do line search at the first iteration of ADMM
                    bla_1 = np.matmul(grad_i.T,grad_j.T)                    
                    Md = np.matmul(M_i,grad_j.T)
                    bla_2 =u*np.matmul(Md.T,S_i)
                    bla_3 = u*np.matmul(Md.T,Md)
                    if bla_3 ==0 :
                        t = gamma
                    else :
#                         t = (bla_1 - bla_2)/bla_3
                          t = bla_1/bla_3
                    print("t",t)
                if line_search:
                    a_k = a_k -t*grad_j.T
                else :
                    a_k = a_k - gamma*grad_j.T
                a_k = np.maximum(a_k,0)
                a_k = np.minimum(a_k,alpha_max)

            '''
            update S_i and rho for each cascades via the closed form formula and the gradient descent respectively
            '''
            for cascs in range(0,M_i.shape[0]) :
                Malpha = np.matmul(M_i[cascs,:],a_k)
                sqrt_delta = math.sqrt((rho[cascs]+u*Malpha)**2+4*u)
                S_i[cascs] = (rho[cascs]+u*Malpha + sqrt_delta)/(2*u)
                if S_i[cascs]<0 :
                    print("Huston Huston we have a probleme")
                if S_i[cascs] ==0 :
                    print("one component of S_i is zero")
                 
                '''
                update rho
                '''
                rho[cascs] = rho[cascs]-u*(S_i[cascs]-Malpha)

            
            k+=1
            '''compute the objective function for node i in iteration k'''
            if k%5 ==0 :
                #print(k)
                obj_i_k = Compute_objective_value_for_one_node(a_k,DAG_C_dic,window,eps,i)
                dic_of_obj_per_node_per_iter[i].append(obj_i_k)
                if Lagrangian :
                    lag_i_k_1 = np.matmul(grad_i,a_k)
                    for c in range(0,M_i.shape[0]):
                        if S_i[c] != 0 :
                            lag_i_k_1 -= math.log(S_i[c])
                            
                    tmp = (np.matmul(M_i,a_k)-S_i).flatten()
                    lag_i_k_1 += np.matmul(rho.T,tmp)
                    lag_i_k_1 += u/2 *np.linalg.norm(tmp)**2
                    lag_per_node_per_iter[i].append(lag_i_k_1)
            
            #Stopping criterion
            if len(dic_of_obj_per_node_per_iter[i])>=2:
                delta = abs(dic_of_obj_per_node_per_iter[i][-1]-dic_of_obj_per_node_per_iter[i][-2])
                if delta<1e-3 :
                    has_converge=True
                    
        A_hat[:,i] = a_k.flatten()
        t_f= time.time()
        time_iter = t_f-t_s
        
        # If more infos is set to true, computes the precision,recall, accuracy and mean square error at each itteration.
        if more_infos :
            (precision,recall,accuracy,mse) = Compute_precision_recall_acc_mse(G_true,A_hat,tol)
            acc_MSE_time.append((accuracy,mse,time_iter))
    return A_hat,dic_of_obj_per_node_per_iter,acc_MSE_time,lag_per_node_per_iter


'''
We tried to use SGD instead of GD in the alpha update. The results were not good enough.
'''
# def NetRate_ADMM_SGD(G_true,C_dic,N,param_ADMM,iter_ADMM_SGD):
        
#     alpha_init,alpha_max,window,iter_GD,iter_ADMM,gamma,u,eps,tol,more_infos = param_ADMM
    
#     dic_of_obj_per_node_per_iter = {}
#     recall_precision_list = []
#     obj_per_node = []
#     acc_MSE_time = []
  
#     A_hat = np.zeros((N,N)) #A_hat will store all the value of the differents alphas.
#     for i in range(0,N) :
#         t_s = time.time()

#         dic_of_obj_per_node_per_iter[i] = []
#         c_index_list = np.random.choice(list(range(0,len(C_dic))),iter_ADMM_SGD,replace = True)

#         '''
#         initialization
#         '''
#         a_k = np.ones(N)*alpha_init
#         a_k[i]=0 # there is no self looping edge
        
#         M_i,T_i = Create_matrix_Mi_and_Ti(C_dic,i,N,window) # TO do : consider to use sparse matrix  

#         S_i = np.matmul(M_i,a_k) # Second variable of the ADMM optimization problem.
#         rho = np.zeros((M_i.shape[0],1)) # Third variable of the ADMM optimization problem.
        
#         has_converge=False
#         k=0
#         #Start iteration of ADMM
#         while has_converge==False and k<iter_ADMM :
#             '''
#             Update alpha using gradient descent
#             '''
#             for j in range(0,len(c_index_list)):
#                 cascade_SGD_index = c_index_list[j]
#                 T_i_c = T_i[cascade_SGD_index]
#                 M_i_c = M_i[cascade_SGD_index]
#                 S_i_c = S_i[cascade_SGD_index]           
#                 rho_c = rho[cascade_SGD_index]           
#                 grad_j = T_i_c - rho_c*M_i_c - u*M_i_c*(np.matmul(M_i_c,a_k)-S_i_c) # + rho_c is not so sure. Perhaps -.
#                 a_k = a_k - gamma*grad_j.T
#                 a_k = np.maximum(a_k,0)
#                 a_k = np.minimum(a_k,alpha_max)

#             '''
#             update S_i and rho for each cascades via the closed form formula and the gradient descent respectively
#             '''
#             for cascs in range(0,M_i.shape[0]) :
#                 Malpha = np.matmul(M_i[cascs,:],a_k)
#                 sqrt_delta = math.sqrt((rho[cascs]+u*Malpha)**2+4*u)
#                 S_i[cascs] = (rho[cascs]+u*Malpha + sqrt_delta)/(2*u)
#                 if S_i[cascs]<0 :
#                     print("Huston Huston we have a probleme")
#                 if S_i[cascs] ==0 :
#                     print("one component of S_i is zero")
                    
#                 rho[cascs] = rho[cascs]-u*(S_i[cascs]-Malpha)

                
#             '''compute the objective function for node i in iteration k'''
#             if k%5 ==0 :
#                 #print(k)
#                 obj_i_k = Compute_objective_value_for_one_node(a_k,C_dic,window,eps,i)
#                 dic_of_obj_per_node_per_iter[i].append(obj_i_k)
            
#             k+=1
            
#             #Stopping criterion
#             if len(dic_of_obj_per_node_per_iter[i])>=2:
#                 delta = abs(dic_of_obj_per_node_per_iter[i][-1]-dic_of_obj_per_node_per_iter[i][-2])
#                 if delta<1e-3 :
#                     has_converge=True
                    
#         A_hat[:,i] = a_k.flatten()
#         t_f= time.time()
#         time_iter = t_f-t_s
        
#         # If more infos is set to true, computes the precision,recall, accuracy and mean square error at each itteration.
#         if more_infos :
#             (precision,recall,accuracy,mse) = Compute_precision_recall_acc_mse(G_true,A_hat,tol)
#             acc_MSE_time.append((accuracy,mse,time_iter))
#     return A_hat,dic_of_obj_per_node_per_iter,acc_MSE_time





def Compute_objective_value_for_one_node(alpha_mat_i,cascade_dic,window,eps,node_i) :
    obj_i = 0
    pot_parent = False
    for cascs in cascade_dic :
        c = cascade_dic[cascs]
        if node_i in c.keys():
            t_i = c[node_i]
            sum_tmp = 0
            for infected in c :
                t_j = c[infected]
                if t_j<t_i:
                    pot_parent = True
                    obj_i += alpha_mat_i[infected]*(t_i-t_j)
                    sum_tmp += alpha_mat_i[infected]
            if sum_tmp<=eps :
                if pot_parent :
#                     print("Attention log ill defined incoming")
                    obj_i -=math.log(eps)
            else :
                obj_i -= math.log(sum_tmp)
        else :
            for j in c.keys():
                t_j = c[j]
                obj_i += alpha_mat_i[j]*(window-t_j)
    return obj_i

def Compute_true_objective_value_per_node(G_true,cascade_dic,window,node_i):
    obj_i = 0
    for cascs in cascade_dic :
        c = cascade_dic[cascs]
        if node_i in c.keys():
            t_i = c[node_i]
            sum_tmp = 0
            pot_parent = False
            for infected in c :
                t_j = c[infected]
                if t_j<t_i :
                    pot_parent = True
                    if (infected,node_i) in G_true.edges():
                        try :
                            alpha_ji = G_true.edges[(infected,node_i)]["weight"][0]
                        except TypeError :
                            alpha_ji = G_true.edges[(infected,node_i)]["weight"]
                        obj_i += alpha_ji*(t_i-t_j)
                        sum_tmp += alpha_ji
            if sum_tmp == 0:
                if pot_parent :
                    print("Oulala")
#                     obj_i -= math.log()
            else :
                obj_i -= math.log(sum_tmp)
        else :
            for j in c :
                if (j,node_i) in G_true.edges():
                    t_j = c[j]
                    try :
                        alpha_ji = G_true.edges[(j,node_i)]["weight"][0]
                    except TypeError :
                        alpha_ji = G_true.edges[(j,node_i)]["weight"]
                    obj_i += alpha_ji*(window-t_j)
    return obj_i