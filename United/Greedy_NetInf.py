import networkx as nx # version 2.2
import math
import operator #to sort elements in a list of tuples


def GetBestEdge(current_prob,last_gain,msort,MIN,dic_of_gain_per_edge,G_star,DAG_Tree_c_dic,cascades_per_edge_dic):
    best_gain = MIN #Assigne value -infinity to the best gain
    best_gain_index = -1
    zero_edge_list = []
    if msort :
        sorted_gain_per_edge_list = sorted(dic_of_gain_per_edge.items(), key=operator.itemgetter(1),reverse=True)
        dic_of_gain_per_edge = dict(sorted_gain_per_edge_list)
        
    key_list = list(dic_of_gain_per_edge.keys())
    attempts = 0
    for index,key_edge in enumerate(dic_of_gain_per_edge) :
        edge = key_edge
        if edge in G_star.edges(): #The edge is already in the network
            continue
        #Computes the marginal gain of adding the edge to the network
        edge_marginal_gain = GetAllCascProb(edge[0],edge[1],DAG_Tree_c_dic,cascades_per_edge_dic)
        dic_of_gain_per_edge[edge] = edge_marginal_gain #Update marginal gain
        if(best_gain<edge_marginal_gain):
            best_gain = edge_marginal_gain
            best_edge = edge
            best_gain_index = index
        attempts +=1 # Needed for sorting later
        
        if (edge not in G_star.edges() and G_star.number_of_edges()>1):
            if(edge_marginal_gain==0) : #Case where there is no improvement in the marginal gain
                zero_edge_list.append(edge)
        
        #Lazy evaluation
        if (index+1 == len(dic_of_gain_per_edge) or best_gain>=dic_of_gain_per_edge[key_list[index+1]]):
            current_prob += best_gain
            if best_gain == 0 :
                return ((-1,-1),current_prob,msort,last_gain,dic_of_gain_per_edge)
            
            del dic_of_gain_per_edge[best_edge] 
            
            for edge_0 in zero_edge_list:
                try :
    #                     if i > best_gain_index :
#                         del dic_of_gain_per_edge[key_list[i-1]]
#                     else :
#                         del dic_of_gain_per_edge[key_list[i]]
                    del  dic_of_gain_per_edge[edge_0]
                except KeyError :
                    print("zero_edge_list",zero_edge_list)
                    print("best edge",best_edge)
                    print("Key_list",key_list)
                    print("Dic of gain",dic_of_gain_per_edge)
                    return
            if len(zero_edge_list)>2:
                attempts = attempts-(len(zero_edge_list)-1)
            msort = (attempts>1)
            last_gain = best_gain
            
            return (best_edge,current_prob,msort,last_gain,dic_of_gain_per_edge)
# This does not make sense for me. But for now I implement the same function they used in their c++ code
def TransProb(DAG, v1,v2):
    global MODEL
    global ALPHA
    global EPS
    eps = EPS
    if( v1 not in DAG.nodes() or v2 not in DAG.nodes()) :
        return eps
    t1 = DAG.nodes[v1]["time"]
    t2 = DAG.nodes[v2]["time"]
    if t1>=t2 :
        return eps
    if MODEL == 0:
        prob = ALPHA*math.exp(-ALPHA*(t2-t1))
    elif MODEL ==1 :
        prob = (ALPHA-1)*math.pow((t2-t1),-ALPHA)
    return prob


def GetAllCascProb(v1,v2,DAG_Tree_c_dic,cascades_per_edge_dic):
    p = 0
    if(v1==-1 and v2 ==-1):
        for c_key in DAG_Tree_c_dic :
            (Tree_c,current_prob_Tc) = UpdateProb(DAG_Tree_c_dic[c_key],v1,v2,False) # Initial Log likelihood for all trees
            p += current_prob_Tc
        return p
    cascade_edge_list = cascades_per_edge_dic[(v1,v2)]
    
    for c_key in cascade_edge_list :
        (Tree_c,current_prob_Tc) = UpdateProb(DAG_Tree_c_dic[c_key],v1,v2,False)
        p +=(current_prob_Tc-DAG_Tree_c_dic[c_key][2]) # marginal gain of adding edge (v1,v2) 
    return p
        
def UpdateProb(DAG_Tree_c_prob,v1,v2,updateProb_bool): 
    DAG_c,Tree_c,current_prob_Tc = DAG_Tree_c_prob
    if(v1 not in Tree_c.nodes() or v2 not in Tree_c.nodes()):
        return (Tree_c,current_prob_Tc)
    if DAG_c.nodes[v1]["time"]>=DAG_c.nodes[v2]["time"] :
        return (Tree_c,current_prob_Tc)
    parent_v2_list = list(Tree_c.predecessors(v2))
    if len(parent_v2_list) == 0:
        parent_v2 = -1 #set an impossible node
    else :
        parent_v2 = parent_v2_list[0]
    p1 = math.log(TransProb(DAG_c, parent_v2,v2))
    p2 = math.log(TransProb(DAG_c,v1,v2))
    if (p1<p2) :
        if(updateProb_bool) :
            if (parent_v2,v2) in Tree_c.edges():
                Tree_c.remove_edge(parent_v2,v2)
            Tree_c.add_edge(v1,v2)
        current_prob_Tc = current_prob_Tc-p1+p2
    return(Tree_c,current_prob_Tc)

def GetBound(edge,cur_loglikelihood,dic_of_gain_per_edge,DAG_Tree_c_dic,cascades_per_edge_dic,G_star) :
    bound = 0
    bounds = []
    for e in cascades_per_edge_dic :
        if (e!=edge and e not in G_star.edges) :
            v1,v2 = e
            e_log_likelihood = GetAllCascProb(v1,v2,DAG_Tree_c_dic,cascades_per_edge_dic) 
            bounds.append(e_log_likelihood) #append the marginal gain
    bounds_sorted = sorted(bounds, reverse = True)
    i = 0
    while(i<G_star.number_of_edges() and i<len(bounds_sorted)) :
        bound +=bounds_sorted[i]
        i+=1
    return bound
            
def GreedyOpt(max_edges,DAG_Tree_c_dic,cascades_per_edge_dic,dic_of_gain_per_edge,G_star,G_true,global_param) :
    global ALPHA
    global MODEL
    global MAX
    global MIN
    global EPS
    (ALPHA,MODEL,MAX,MIN,EPS,compare_ground_truth,boundOn) = global_param
    
    current_log_likelihood = GetAllCascProb(-1,-1,DAG_Tree_c_dic,cascades_per_edge_dic)
    last_gain = MAX
    msort = False
    
    #Needed if compare ground truth is True
    precision_list = []
    recall_list = []
    precision_recall = []
    
    #In case boundOn is true
    bound = 0 
    edge_info_dic = {}
    
    k = 0
    while (k<max_edges and len(dic_of_gain_per_edge)>0):
        prev = current_log_likelihood
        print("itteration : ",k)
        (best_edge,current_log_likelihood,msort,last_gain,dic_of_gain_per_edge) = GetBestEdge(current_log_likelihood,
                                                                                    last_gain,
                                                                                    msort,
                                                                                    MIN,
                                                                                    dic_of_gain_per_edge,
                                                                                    G_star,
                                                                                    DAG_Tree_c_dic,
                                                                                    cascades_per_edge_dic)
#         print("Best edge is ",best_edge)
        if best_edge == (-1,-1): #No more edges can be added to G_star
            break
            
        if (compare_ground_truth) :
            precision = 0
            recall = 0
            if (len(precision_recall)>1):
                precision,recall = precision_recall[-1]
            if best_edge in G_true.edges():
                recall += 1
            else :
                precision +=1
            precision_recall.append((precision,recall))
            
        G_star.add_edge(best_edge[0],best_edge[1])
        k+=1
        bound = 0
        if (boundOn) :
            bound = GetBound(best_edge, prev,dic_of_gain_per_edge,DAG_Tree_c_dic,cascades_per_edge_dic,G_star)
        edge_info_dic[best_edge] = (last_gain,bound) 
        
        #Localized update
        cascade_local = cascades_per_edge_dic[best_edge]
        for c in cascade_local :
            Tree_c,current_prob_Tc = UpdateProb(DAG_Tree_c_dic[c],best_edge[0],best_edge[1],True)
            DAG_Tree_c_dic[c] = (DAG_Tree_c_dic[c][0],Tree_c,current_prob_Tc)

    
    if (compare_ground_truth) :
        for pair in precision_recall :
            precision_i = 1- pair[0]/(pair[0]+pair[1]) # Which fraction of edge in G_k are also in G*
            recall_i = pair[1]/G_true.number_of_edges() #Which fractuin of edges in G* are also in G_k
            precision_list.append(precision_i)
            recall_list.append(recall_i) 
            
    return G_star,precision_list,recall_list,edge_info_dic