# GAClust

SYNOPSIS

GAGraphClust is a method to cluster graphs to optimize modularity of the graph. It uses a genetic algorithm
with various operations such as cross-over, mutation and elitist selection. Individual parameters for each
operation are defined in the following sections.

PARAMETERS

filename: The filename of the graph input file.

The first line of this file begins with a hash ‘#’ followed by the
number of nodes and edges in the following format:
\#n=34,m=78
Each following line is tab-delimited and contains the two end-points of each edge:
#1 0
#2 0
#...
#33 32
In general, if there are m edges in the graph, there should be exactly m + 1 lines in the file. All
graphs are undirected and edges are repeated only once (e.g. in the above example, there will not
be a second line with ‘0 1’ since this is the same as the first line for undirected graphs).

maximum number of clusters: For initial generation of clusters, a maximum number of clusters needs to be suggested.
This parameter is used to limit computational requirement of the algorithm. This parameter is optional. Default value is
5.

maximum number of iterations: iterations are converged if the best individual will not change more than 20 iterations.
To guarantee exit of program, a maximum number of iterations is defined. This parameter is optional. Default value is
100.

population size: the number of individuals in each population. This parameter is optional. Default value is 1000.

elitist selection parameter: the ratio of the individuals with high fitness carried over to next generations.
This parameter is optional. Default value is 0.25.

mutation rate: the rate of mutation for each element in the individual vectors. This parameter is optional. Default value is 0.1

USAGE

Usage: python setup.py -i <filename> [-N <MaxNumClass>] [-n <MaxNumIteration>] [-a <PopulationSize>] [-b <ElitismParameter>] [-g <MutationRate>]
