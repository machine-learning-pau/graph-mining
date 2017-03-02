import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans
#########
#choices is a list of tuples (c,w)
#the method returns a choice of c based on a proba proportional to w
def weighted_choice(choices):
    total = sum(w for (c, w) in choices)#compute the sum of w
    r = rd.uniform(0, total)#choose a random number in [0,total]
    upto = 0
    for (c, w) in choices:#select the first time upto+w is greater than r
        if upto + w > r:
            return c
        upto += w
    assert False, "Shouldn't get here"

# example of PA(m)
m=100
result = [(0,1)]#first edge
degrees = [1,1]#first degrees
for k in range(2,m):
    liste = zip(range(len(degrees)),degrees)#construct the list of tuple (node,degree)
    vertex = weighted_choice(liste)#choice of the vertex to link with
    result.append((k,vertex))#new vertex k linked with vertex
    degrees[vertex]+=1#add 1 to the deggree of vertex
    degrees.append(1)#add a degree 1 for vertex k

G=nx.Graph(result)#construct the graph with edgelist result
nx.draw(G)#draw the graph
plt.show()#plot the graph

#Spectral clustering algorithm

A = nx.adjacency_matrix(G)#compute the adjacency matrix of G (SciPy sparse matrix)
A = A.todense()#transform the SciPy sparse matrix into a dense matrix
L = np.diag(G.degree().values())-A#compute the Laplacian matrix L

w,v = np.linalg.eig(L)#compute the eigendecomposition of L 

w = np.real(w)#transform complex values into real values 
v = np.real(v)#transform complex values into real values 

idx = w.argsort()
w_sorted = w[idx]#sort in a nondecreasing order the eigenvalues
v_sorted = v[:,idx]#and the corresponding eigenvectors

k = 5#number of clusters
kmeans = KMeans(k, n_init=10)#instance of kmeans clustering algorithm

clustering = kmeans.fit(v_sorted[:,:k])#compute the clustering over the k first eigenvectors

clustering.labels_#return the labels of each node



