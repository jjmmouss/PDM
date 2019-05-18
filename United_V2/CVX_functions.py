import networkx as nx # version 2.2
import cvxpy as cp
import numpy as np
import time

'''
Create the matrices for the optimization in the cas of the exponential model

Input :
    - N : Number of vertices of the graph we want to infer (i.e number of node on which data were collected)
    - cascades_dic : dictionary containing the dictionary per each cascade
    - window : Time during which we recorde the cascade
Output :
    - (A_pot,A_bad) : two matrices corresponding to the values (without the mult. by alpha) of the log survival funvtion in respectivly the infected case and not infected case.
    - cascade_per_nodes : list of cascade in which a node is present in
'''
def Create_matrices(N,cascades_dic,window):
    cascades_per_nodes = {}
    #A_pot and A_bad correspond to the values that are summed in repectivly Psi_2 and Psi_1 (without the multiplication by alpha) in the NetRate paper
    A_pot = np.zeros((N,N)) # more or less corresponds to Psi 2 in the paper
    A_bad = np.zeros((N,N)) # more or less corresponds to Psi 1 in the paper
    for c in cascades_dic :
        cascade = cascades_dic[c]
        for i in cascade:
            try:
                cascades_per_nodes[i].append(c)
            except KeyError :
                cascades_per_nodes[i] = [c]
            t_v1 = cascade[i]
            for infected in cascade :
                t_v2 = cascade[infected]
                if t_v2<t_v1 : #potential parent
                    A_pot[infected,i] += t_v1-t_v2 #Since log(S(t_v2|t_v1;alpha_v1,v2)) = alpha_v1,v2 * (t_v2-t_v1)

        for node in range(0,N) :
            if node not in cascade.keys() : # Meaning this node was not infected during cascade c
                for node_infected in cascade.keys():
                    t_infected = cascade[node_infected]
                    A_bad[node_infected,node] += window - t_infected #survival function in the not infected case
    return (A_pot,A_bad),cascades_per_nodes

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
    #         print("For edge " + str(edge)+ " the true weight is : " + str(true_alpha))
    #         print("For edge " + str(edge)+ " the approx weight is : " + str(approx_alpha))
    try :
        mse = mse/G_true.number_of_edges()
    except ZeroDivisionError:
        print ("There is no correct edge")
    return (precision,recall,accuracy,mse)

'''
Main function, solving the optimization problem with CVX. Returns the matrix A_hat containing the value of each alphas.

Input :
    -N : number of nodes that we have data on.
    -matrix_list : (A_pot,A_bad) matrices computed in the create matrix function.
    -cascades_per_node : dictionary in which each index (corresponding to a node) has a list of all cascade in which it is present.
    -cascades_dic : dictionary in which each index, k, is a dictionary containg all nodes infected and their infection time during cascade k.

Output :
    -A_hat : matrix of size N X N containing the vaues of alpha for each potential edge (i,j).
    -total_obj : value of the objective function.
'''
def Infer_Network_edges(N,matrix_list,nb_cascades_per_node,cascades_dic,more_infos):
    A_hat = np.zeros((N,N))
    A_pot = matrix_list[0]
    A_bad = matrix_list[1]
    total_obj = 0
    Acc_MSE_time = []
    
    for i in range(0,N):
        print("For node ", i)
        t_s = time.time()
        try :
            if len(nb_cascades_per_node[i])== 0:
                A_hat[:,i] = 0
                continue
        except :
            A_hat[:,i] = 0
            continue
            
        #Start of cvxnp
        a_hat = cp.Variable(N) # is alpha_j,i restricted to one column (i.e one node) hence good for // computing.
        t_hat = cp.Variable(len(nb_cascades_per_node[i]))
        obj = 0
        constr = []   
        for j in range(0,N):
            if A_pot[j,i]>0:
                obj = -a_hat[j]*(A_pot[j,i]+A_bad[j,i]) + obj
            elif A_pot[j,i]==0:
                constr.append(a_hat[j]==0)

        c_act = 0
        for c in cascades_dic :
            tmp = 0
            cascade = cascades_dic[c]
            has_parent = False
            if i in cascade.keys():
                t_i = cascade[i]
                for infected in cascade.keys():
                    t_j = cascade[infected]
                    if t_j<t_i:
                        has_parent = True
                        tmp = tmp + a_hat[infected]
                if has_parent :
                    constr.append(t_hat[c_act]==tmp)
                    obj = obj + cp.log(t_hat[c_act])
                    c_act += 1

        constr.append(a_hat>=0)
        objf = cp.Maximize(obj)
        prob = cp.Problem(objf,constr)
        try :
            result = prob.solve()
        except cp.SolverError:
            print("Use another solver")
            result = prob.solve(solver = cp.SCS)
            print(prob.status)
        total_obj -= prob.value # this is coded for a maximization problem but we would like to compare with the other algos that solve the minimization version.
        A_hat[:,i] = a_hat.value
        t_f = time.time()
        time_algo = t_f-t_s
        if more_infos :
            (precision,recall,accuracy,mse)= Compute_Acc(ground_truth,A_hat,tol)
            Acc_MSE_time.append((accuracy,mse,time_algo))
    return A_hat,total_obj,Acc_MSE_time