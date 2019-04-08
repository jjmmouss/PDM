import networkx as nx # version 2.2
import cvxpy as cp
import numpy as np

'''
Create the matrices for the optimization in the cas of the exponential model

Input :
    - ground_truth : Directed graph (we use ground truth only to itterate over the nodes, so G_star would do the same)
    - cascades_DAG_dic : dictionary containing the directed graph per each cascade
    - window : Time during which we recorde the cascade
Output :
    - (A_pot,A_bad) : two matrices corresponding to the values (without the mult. by alpha) of the log survival funvtion in respectivly the infected case and not infected case.
    - num_cascade_per_nodes : list of number of cascade a node is present in
'''
def Create_matrices(ground_truth,cascades_DAG_dic,window):
    num_cascade_per_nodes = np.zeros((1,ground_truth.number_of_nodes()))
    #A_pot and A_bad correspond to the values that are summed in repectivly Psi_2 and Psi_1 (without the multiplication by alpha) in the NetRate paper
    A_pot = np.zeros((ground_truth.number_of_nodes(),ground_truth.number_of_nodes())) # more or less corresponds to Psi 2 in the paper
    A_bad = np.zeros((ground_truth.number_of_nodes(),ground_truth.number_of_nodes())) # more or less corresponds to Psi 1 in the paper
    for c in cascades_DAG_dic :
        DAG = cascades_DAG_dic[c]

        for i in DAG.nodes():
            num_cascade_per_nodes[0,i] += 1
            t_v1 = DAG.nodes[i]["time"]
            parents = list(DAG.predecessors(i))
            if len(parents)==0:
                continue
            for j in parents :
                t_v2 = DAG.nodes[j]["time"]
                A_pot[j,i] += t_v1-t_v2 #Since log(S(t_v2|t_v1;alpha_v1,v2)) = alpha_v1,v2 * (t_v2-t_v1)
                if (t_v1-t_v2)<=0:
                    print("Error delta time neg")
        for node in ground_truth.nodes() :
            if node not in DAG.nodes() : # Meaning this node was not infected during cascade c
                for node_infected in DAG.nodes():
                    t_infected = DAG.nodes[node_infected]["time"]
                    A_bad[node_infected,node] += window - t_infected #survival function in the not infected case
    return (A_pot,A_bad),num_cascade_per_nodes

def Create_matrices_ADMM(ground_truth,cascades_DAG_dic,window):
    num_cascade_per_nodes = {}
    #A_pot and A_bad correspond to the values that are summed in repectivly Psi_2 and Psi_1 (without the multiplication by alpha) in the NetRate paper
    A_pot = np.zeros((ground_truth.number_of_nodes(),ground_truth.number_of_nodes())) # more or less corresponds to Psi 2 in the paper
    A_bad = np.zeros((ground_truth.number_of_nodes(),ground_truth.number_of_nodes())) # more or less corresponds to Psi 1 in the paper
    for c in cascades_DAG_dic :
        DAG = cascades_DAG_dic[c]

        for i in DAG.nodes():
            try:
                num_cascade_per_nodes[i].append(c)
            except KeyError :
                num_cascade_per_nodes[i] = [c]
            t_v1 = DAG.nodes[i]["time"]
            parents = list(DAG.predecessors(i))
            if len(parents)==0:
                continue
            for j in parents :
                t_v2 = DAG.nodes[j]["time"]
                A_pot[j,i] += t_v1-t_v2 #Since log(S(t_v2|t_v1;alpha_v1,v2)) = alpha_v1,v2 * (t_v2-t_v1)
                if (t_v1-t_v2)<=0:
                    print("Error delta time neg")
        for node in ground_truth.nodes() :
            if node not in DAG.nodes() : # Meaning this node was not infected during cascade c
                for node_infected in DAG.nodes():
                    t_infected = DAG.nodes[node_infected]["time"]
                    A_bad[node_infected,node] += window - t_infected #survival function in the not infected case
    return (A_pot,A_bad),num_cascade_per_nodes

def Infer_Network_edges(ground_truth,matrix_list,nb_cascades_per_node,cascades_DAG_dic):
    A_hat = np.zeros((ground_truth.number_of_nodes(),ground_truth.number_of_nodes()))
    A_pot = matrix_list[0]
    A_bad = matrix_list[1]
    total_obj = 0
    for i in ground_truth.nodes():
        print("For node ", i)
        if nb_cascades_per_node[0,i]== 0:
            A_hat[:,i] = 0
            continue
        #Start of cvxnp
        a_hat = cp.Variable(ground_truth.number_of_nodes()) # is alpha_j,i restricted to one column (i.e one node) hence good for // comput.
        t_hat = cp.Variable(int(nb_cascades_per_node[0,i]))
        obj = 0
        constr = []   
        for j in ground_truth.nodes():
            if A_pot[j,i]>0:
                obj = -a_hat[j]*(A_pot[j,i]+A_bad[j,i]) + obj
            elif A_pot[j,i]==0:
                constr.append(a_hat[j]==0)

        c_act = 0
        for c in cascades_DAG_dic :
            tmp = 0
            DAG = cascades_DAG_dic[c]
            if i in DAG.nodes():
                i_parents = list(DAG.predecessors(i))
                if len(i_parents)>0:
                    for papa in i_parents :
                        tmp = tmp + a_hat[papa]
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
        print(prob.value)
        total_obj += prob.value
#         print(a_hat.value)
        A_hat[:,i] = a_hat.value
    return A_hat,total_obj