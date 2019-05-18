import sys

def Generate_param():
    window = 10
    beta = 1 # used for the construction of the cascades, probability of the edge to transmit the infection
    eps = 0.0005
    alpha_max = 10 #bound alpha from above
    alpha_init = 0.01 #initial value of the infectious rates
    iter_ADMM = 50 #number of itteration for 1 node in the ADMM method. This is a parameter to tune
    iter_GD_ADMM = 1000 #number of iterations of the gradient descent
    gamma_ADMM = 0.000005 # Learning rate of the GD for alpha (maybe too small)
    u = 4 # used for the gradient descent of rho and as a penalizer and the constrain


    gamma_SGD = 0.0005
    tol = 1e-2
    K = 15000 #number of itterations of SG

    EPS = 1e-64 #zero machine
    ALPHA = 1.0 #Incubation parameter (for exp and power law)
    MODEL = 0 # 0 = exp law, 1 = power law (power law is not  implemented yet) When constructing the underlying network specify that we use the exponential law
    MAX = sys.float_info.max #Max value of a float in python
    #MIN = sys.float_info.min #Min value of a float in python
    MIN = -MAX

    #(works only if groundtruth is available)
    #When set to True (especially boundOn) it slow down greatly the computation
    compare_groud_truth = False # If set to True outputs some aditional information (precision and recall of the algo)
    boundOn = False


    greedy_global_param = (ALPHA,MODEL,MAX,MIN,EPS,compare_groud_truth,boundOn)
    ADMM_param = (alpha_init,alpha_max,window,iter_GD_ADMM,iter_ADMM,gamma_ADMM,u,eps,tol)
    SGD_param = (alpha_init,alpha_max,K,gamma_SGD,eps,window)
    return greedy_global_param,SGD_param,ADMM_param