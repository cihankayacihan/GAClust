#!/usr/bin/env python

''' Code for solving graph clustering with genetic algorithm
	Author: Cihan Kaya
'''

import os
import sys, getopt
import numpy as np
import argparse

def calcAdj(graph, nodes):
	''' This function calculates the adjacency matrix of a given graph 
	'''
	adj = np.zeros((nodes, nodes))
	for i in range(len(graph)):
		adj[graph[i][0],graph[i][1]]=1.
		adj[graph[i][1],graph[i][0]]=1.

	return adj

def calcDegrees(graph, nodes):
	''' This function calculates the degree of each node on the graph '''
	degrees=np.zeros(nodes,dtype=np.int)
	for i in range(nodes):
		degrees[i]=list(np.ravel(graph)).count(i)
	return degrees
	

def objective(graph, adj, cluster, degrees, nodes, edges):
	''' Objective function of the system which is modularity value from Girvan et. al, 2002, PNAS '''
	objective=0.
	for i in range(nodes-1):
		for j in range(i+1,nodes):
			objective += ((adj[i,j]-degrees[i]*degrees[j]/(2.*edges))*(1.-(cluster[i]!=cluster[j])))
	return objective

def GAClust(graph,degrees,nodes,edges,adj,N,N_iter,alpha,beta,gamma):
	''' This function calculates the optimal clustering to maximize the modularity of the given graph with genetic algorithm. '''

	N = int(N) # maximum number of clusters
	N_iter = int(N_iter) # number of maximum iterations
	alpha = int(alpha) # Population size
	beta = float(beta) # Elitism/Selection parameter
	gamma = float(gamma) # mutation rate

	# Initialization of clusters
	cluster = np.random.randint(N, size=(alpha,nodes))
	# Initialization of objective functions
	objectives = np.zeros(alpha)
	# Number of elite individuals
	ne = int(alpha * beta)
	# Number of crossovers
	nc = (alpha - ne)/2
	# Two separate populations are created. Elite and Nonelite
	pop1 = np.zeros((ne,nodes),dtype=np.int)
	pop2 = np.zeros((alpha-ne,nodes),dtype=np.int)
	# initial best objective function
	best_obj = 0
	# initial best solution
	best_sol = np.zeros(nodes, dtype=np.int)
	m=0 # iteration termination value 
	sys.stdout.write("Running ")
	sys.stdout.write("[%s]" % (" " * N_iter))
	sys.stdout.flush()
	sys.stdout.write("\b" * (N_iter+1))
	
	# fitness function evaluation for the first time
	for j in range(cluster.shape[0]):
		objectives[j] = objective(graph,adj,cluster[j],degrees,nodes,edges)

	for i in range(N_iter):
		# based on fitness function, selection of elite class
		with np.errstate(invalid='ignore'):
			# normalization of rates
			rates = (objectives-min(objectives))/(np.sum(objectives-min(objectives)))
			# selection of top ne of population
			index_elite = np.random.choice(alpha,ne,p=rates)
		pop1=cluster[index_elite]
		# based on fitness function of elite class, cross over operations
		rate_elite = rates[index_elite]/sum(rates[index_elite])
		k=0 
		for j in range(int(nc)):
			with np.errstate(invalid='ignore'):
				Xa = cluster[np.random.choice(index_elite,p=rate_elite)]
				Xb = cluster[np.random.choice(index_elite,p=rate_elite)]
			changes = np.random.randint(nodes, size=np.random.randint(nodes))
			Xc = Xa.copy()
			Xd = Xb.copy()
			for l in range(len(changes)):
				Xc[changes[l]]=Xb[changes[l]]
				Xd[changes[l]]=Xa[changes[l]]
			pop2[k]=Xc
			pop2[k+1]=Xd
			k=k+2
		# for new generated individuals mutation operations are performed
		for j in range(int(nc)):
			indice = np.random.randint(alpha-ne)
			new = np.zeros(nodes, dtype=np.int)
			for k in range(nodes):
				new[k]=(pop2[indice,k]+np.random.randint(int(N))*(np.random.rand()<gamma))%N
		# concatanate elite population with newly created population
		cluster = np.concatenate((pop1,pop2),axis=0)
		# new fitness function calculation
		for j in range(cluster.shape[0]):
			objectives[j] = objective(graph,adj,cluster[j],degrees,nodes,edges)
		# if it is better than last seen best solution, update best solution
		if max(objectives)>best_obj:
			m=0
			best_obj = max(objectives)
			best_sol[:] = cluster[np.argmax(objectives==best_obj)]
		# if the best solution stays for 20 iterations return best solution
		if m==20:
			return best_sol
		sys.stdout.write("-")
		sys.stdout.flush()
	sys.stdout.write("\n")
	return best_sol


def main(argv):
	N_iter = 100
	N =5 
	alpha = 1000
	beta = 0.25
	gamma = 0.1
	filename = None
	try:
		opts, args = getopt.getopt(argv,"hi:N:n:a:b:g:",["filename=","NumClass=","N_iter=","alpha=","beta=","gamma="])
	except getopt.GetoptError:
		print 'Usage: python GAGraphClust.py -i <filename> [-N <MaxNumClass>] [-n <MaxNumIteration>] [-a <PopulationSize>] [-b <ElitismParameter>] [-g <MutationRate>]'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'Usage: python GAGraphClust.py -i <filename> [-N <MaxNumClass>] [-n <MaxNumIteration>] [-a <PopulationSize>] [-b <ElitismParameter>] [-g <MutationRate>]'
			sys.exit()
		elif opt in ("-i", "--filename"):
		 	filename = arg
		elif opt in ("-N", "--NumClass"):
			N = arg
		elif opt in ("-n", "--N_iter"):
			N_iter = arg
		elif opt in ("-a", "--alpha"):
			alpha = arg
		elif opt in ("-b", "--beta"):
			beta = arg
		elif opt in ("-g", "--gamma"):
			gamma = arg
	if filename == None:
		print 'Usage: python GAGraphClust.py -i <filename> [-N <MaxNumClass>] [-n <MaxNumIteration>] [-a <PopulationSize>] [-b <ElitismParameter>] [-g <MutationRate>]'
                sys.exit(2)

	with open(filename, 'r') as f:
		nodes, edges = f.readline().split(',')
	# parse input from graph and determine number of nodes and edges 
	nodes=int(filter(type(nodes).isdigit,nodes))
	edges=int(filter(type(edges).isdigit,edges))
	data=np.loadtxt(filename, dtype=np.int)
	# calculate adjacency matrix
	adj=calcAdj(data,nodes)
	# precalculate degrees of each nodes
	degrees = calcDegrees(data, nodes)
	# call genetic algorithm to get the best cluster
	best_clust = GAClust(data,degrees,nodes,edges,adj,N,N_iter,alpha,beta,gamma)
	# get the fitness function value of best cluster 
	obj = objective(data, adj, best_clust, degrees, nodes, edges)

	# print output to the standart out
	clusters = np.unique(best_clust)
	members = []
	print "#Modularity " + str(obj)
	for i in range(len(clusters)):
		members.append([])
	for i in range(nodes):
		members[np.where(best_clust[i]==clusters)[0][0]].append(i)

	for i in range(len(members)):
		print "Module " + str(i+1) + ":\t",
		for j in range(len(members[i])):
			print str(members[i][j]) + "\t",
		print ""

if __name__ == "__main__":
   main(sys.argv[1:])





