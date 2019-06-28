# PDM

This repository contains the code for the algorithm of NetInf, NetRate, and ARate described in the master thesis "Influence Networks and How to Find Them".

The main folder is United_V2. It contains all the algorithm in their final shape.

Inside United_V2 there is :

- United_algos_V2.ipynb it is the python notebook executing all the algorithms and outputing plots and graphs.
-CVX_functions.py contains the functions needed for NetRate.
-Cascade_generation_functions_NetRate.py contains the underlying network generation and the cascades propagations in for the algorithm NetRate, InfoPath and ARate. It is needed to generate our data and to compare our solution to the true network.
-Greedy_NetInf and Init_NetInf contains the functions for the NetInf algorithm. The graphs are stored differently as for the others algorithms.
